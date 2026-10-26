//! Per-key stream isolation for `fetch`.
//!
//! By default every stream a `TorClient` opens may share circuits, so requests
//! made seconds apart leave the same exit and can be linked by whoever answers
//! them. A caller that passes `anonOptions.isolationKey` gets one Arti
//! isolation group per distinct key: requests with the same key may share
//! circuits; requests with different keys never do. Requests without a key
//! form one more group of their own.
//!
//! The key is the caller's opaque label (an anon-rpc host might derive it from
//! the account a request is about). It is never logged or sent anywhere.
//!
//! A group can also move to a fresh token. Arti keeps handing out a circuit
//! whose exit failed to open a stream until the circuit expires, so `fetch`
//! moves the group off it after such a failure (see [`connect_in_group`]).

use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::rc::Rc;

use arti_client::{ErrorKind, HasKind, IsolationToken};
use tracing::info;

/// Longest accepted key. Keys are labels, not payloads.
pub(crate) const MAX_KEY_LEN: usize = 256;

/// Bound on remembered keys. Past it the map starts over: a key seen again gets
/// a fresh group, which is still isolated from every other key.
const MAX_KEYS: usize = 4096;

/// Why an isolation key was refused.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum KeyError {
    Empty,
    TooLong,
}

/// Maps isolation keys to Arti isolation groups for one client.
pub(crate) struct IsolationGroups {
    /// The group of requests made without a key.
    unkeyed: IsolationToken,
    groups: HashMap<String, IsolationToken>,
}

impl Default for IsolationGroups {
    fn default() -> Self {
        Self {
            unkeyed: IsolationToken::new(),
            groups: HashMap::new(),
        }
    }
}

impl IsolationGroups {
    /// The isolation group for `key`, created on first use.
    pub(crate) fn token_for(&mut self, key: &str) -> Result<IsolationToken, KeyError> {
        if key.is_empty() {
            // An empty key is almost certainly a caller bug; sharing circuits
            // silently would be the unsafe reading of it.
            return Err(KeyError::Empty);
        }
        if key.len() > MAX_KEY_LEN {
            return Err(KeyError::TooLong);
        }
        if let Some(token) = self.groups.get(key) {
            return Ok(*token);
        }
        Ok(self.insert_fresh(key))
    }

    /// The isolation group of requests made without a key.
    pub(crate) fn unkeyed(&self) -> IsolationToken {
        self.unkeyed
    }

    /// Move the group of `key` (the unkeyed group for `None`) off `stale` to a
    /// fresh token, and return the group's token.
    ///
    /// Requests that failed together on one circuit all report the same
    /// `stale` token; only the first moves the group, and the rest retry on the
    /// token it chose instead of each building a circuit of its own.
    pub(crate) fn replace(&mut self, key: Option<&str>, stale: IsolationToken) -> IsolationToken {
        let current = match key {
            None => &mut self.unkeyed,
            Some(key) => match self.groups.get_mut(key) {
                Some(token) => token,
                // Forgotten since the request started: it gets a fresh group anyway.
                None => return self.insert_fresh(key),
            },
        };
        if *current == stale {
            *current = IsolationToken::new();
        }
        *current
    }

    fn insert_fresh(&mut self, key: &str) -> IsolationToken {
        if self.groups.len() >= MAX_KEYS {
            self.groups.clear();
        }
        let token = IsolationToken::new();
        self.groups.insert(key.to_owned(), token);
        token
    }
}

/// One request's isolation group: its token, and the means to move the group
/// to a fresh one.
pub(crate) struct Group {
    groups: Rc<RefCell<IsolationGroups>>,
    key: Option<String>,
    token: IsolationToken,
}

impl Group {
    /// The group for a request with `key`, or the unkeyed group for `None`.
    pub(crate) fn new(
        groups: &Rc<RefCell<IsolationGroups>>,
        key: Option<String>,
    ) -> Result<Self, KeyError> {
        let token = match &key {
            Some(key) => groups.borrow_mut().token_for(key)?,
            None => groups.borrow().unkeyed(),
        };
        Ok(Self {
            groups: Rc::clone(groups),
            key,
            token,
        })
    }

    pub(crate) fn token(&self) -> IsolationToken {
        self.token
    }

    pub(crate) fn is_keyed(&self) -> bool {
        self.key.is_some()
    }

    /// Move the group off its current token (see [`IsolationGroups::replace`]),
    /// so this request's retry and requests that start later get a circuit
    /// other than the one this request failed on.
    pub(crate) fn move_to_fresh_token(&mut self) {
        self.token = self
            .groups
            .borrow_mut()
            .replace(self.key.as_deref(), self.token);
    }
}

/// Why [`connect_in_group`] gave up.
pub(crate) enum ConnectError<E> {
    Failed(E),
    Aborted,
}

/// Open a stream with `connect` on a circuit of `group`.
///
/// Arti keeps giving out a circuit after a stream failed to open on it, until
/// the circuit expires (ten minutes after first use for a shared one), so one
/// bad exit can fail every request in a group for that long. Each time the far
/// end fails to open the stream, move the group to a fresh token, so requests
/// that start later get another circuit, and try once more after the first
/// failure unless `aborted`. This approximates C tor's
/// `mark_circuit_unusable_for_new_conns`: requests that already hold the old
/// token can still reach the old circuit, which expires on its own schedule.
pub(crate) async fn connect_in_group<T, E, Fut>(
    group: &mut Group,
    mut connect: impl FnMut(IsolationToken) -> Fut,
    aborted: impl Fn() -> bool,
) -> Result<T, ConnectError<E>>
where
    E: HasKind + std::error::Error,
    Fut: Future<Output = Result<T, E>>,
{
    let mut retried = false;
    loop {
        match connect(group.token()).await {
            Err(e) if exit_failed(e.kind()) => {
                group.move_to_fresh_token();
                if retried {
                    return Err(ConnectError::Failed(e));
                }
                info!(
                    "Exit failed to open a stream ({}); retrying on a fresh circuit",
                    crate::error::with_sources(&e)
                );
                if aborted() {
                    return Err(ConnectError::Aborted);
                }
                retried = true;
            }
            result => return result.map_err(ConnectError::Failed),
        }
    }
}

/// Whether an error of `kind` says the far end of the circuit failed to open
/// a stream: it timed out (an exit, or an onion service's rendezvous), or the
/// exit answered with an END reason that C tor retries on another circuit. A
/// refused connection is the destination's answer, not the exit's, so it is
/// not retried.
fn exit_failed(kind: ErrorKind) -> bool {
    use ErrorKind as EK;
    matches!(
        kind,
        // Arti's own connect timeout; it reports this kind, not `ExitTimeout`.
        EK::RemoteNetworkTimeout
            // END reasons TIMEOUT, MISC, RESOLVEFAILED, NOROUTE, EXITPOLICY,
            // and HIBERNATING or RESOURCELIMIT.
            | EK::ExitTimeout
            | EK::RemoteStreamError
            | EK::RemoteHostResolutionFailed
            | EK::RemoteNetworkFailed
            | EK::ExitPolicyRejected
            | EK::RelayTooBusy
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_key_same_group_different_keys_different_groups() {
        let mut groups = IsolationGroups::default();
        let a1 = groups.token_for("account-a").unwrap();
        let a2 = groups.token_for("account-a").unwrap();
        let b = groups.token_for("account-b").unwrap();
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }

    #[test]
    fn groups_are_per_client() {
        let mut one = IsolationGroups::default();
        let mut two = IsolationGroups::default();
        assert_ne!(one.token_for("k").unwrap(), two.token_for("k").unwrap());
    }

    #[test]
    fn rejects_empty_and_oversized_keys() {
        let mut groups = IsolationGroups::default();
        assert_eq!(groups.token_for(""), Err(KeyError::Empty));
        assert_eq!(
            groups.token_for(&"x".repeat(MAX_KEY_LEN + 1)),
            Err(KeyError::TooLong)
        );
        assert!(groups.token_for(&"x".repeat(MAX_KEY_LEN)).is_ok());
    }

    #[test]
    fn starting_over_still_isolates() {
        let mut groups = IsolationGroups::default();
        let first = groups.token_for("k").unwrap();
        for i in 0..MAX_KEYS {
            groups.token_for(&format!("filler-{i}")).unwrap();
        }
        let again = groups.token_for("k").unwrap();
        assert_ne!(
            first, again,
            "a forgotten key gets a fresh group, never another key's"
        );
    }

    #[test]
    fn unkeyed_requests_share_a_group_of_their_own() {
        let mut groups = IsolationGroups::default();
        let k = groups.token_for("k").unwrap();
        assert_eq!(groups.unkeyed(), groups.unkeyed());
        assert_ne!(groups.unkeyed(), k);
    }

    #[test]
    fn replace_moves_a_group_once_per_stale_token() {
        let mut groups = IsolationGroups::default();
        let stale = groups.token_for("k").unwrap();
        let fresh = groups.replace(Some("k"), stale);
        assert_ne!(fresh, stale);
        assert_eq!(groups.token_for("k").unwrap(), fresh);
        // A second request that failed on the same circuit joins the first.
        assert_eq!(groups.replace(Some("k"), stale), fresh);

        let stale = groups.unkeyed();
        let fresh = groups.replace(None, stale);
        assert_ne!(fresh, stale);
        assert_eq!(groups.unkeyed(), fresh);
        assert_eq!(groups.replace(None, stale), fresh);
    }

    #[test]
    fn replace_leaves_other_groups_alone() {
        let mut groups = IsolationGroups::default();
        let a = groups.token_for("a").unwrap();
        let unkeyed = groups.unkeyed();
        let b = groups.token_for("b").unwrap();
        groups.replace(Some("b"), b);
        assert_eq!(groups.token_for("a").unwrap(), a);
        assert_eq!(groups.unkeyed(), unkeyed);
    }

    #[test]
    fn group_handle_moves_the_shared_group() {
        let groups = Rc::new(RefCell::new(IsolationGroups::default()));
        let mut first = Group::new(&groups, Some("k".into())).unwrap();
        let second = Group::new(&groups, Some("k".into())).unwrap();
        assert_eq!(first.token(), second.token());
        first.move_to_fresh_token();
        assert_ne!(first.token(), second.token());
        assert_eq!(
            Group::new(&groups, Some("k".into())).unwrap().token(),
            first.token()
        );
        assert!(matches!(
            Group::new(&groups, Some(String::new())),
            Err(KeyError::Empty)
        ));
    }

    #[derive(Debug)]
    struct Failure(ErrorKind);

    impl std::fmt::Display for Failure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::error::Error for Failure {}

    impl HasKind for Failure {
        fn kind(&self) -> ErrorKind {
            self.0
        }
    }

    /// Run `connect_in_group` on the unkeyed group, with each attempt taking
    /// the next of `outcomes`. Returns the tokens the attempts used, the
    /// result, and the group's token afterwards.
    fn run(
        outcomes: Vec<Result<(), ErrorKind>>,
        aborted: bool,
    ) -> (
        Vec<IsolationToken>,
        Result<(), ConnectError<Failure>>,
        IsolationToken,
    ) {
        let groups = Rc::new(RefCell::new(IsolationGroups::default()));
        let mut group = Group::new(&groups, None).unwrap();
        let mut outcomes = outcomes.into_iter();
        let mut used = Vec::new();
        let result = futures::executor::block_on(connect_in_group(
            &mut group,
            |token| {
                used.push(token);
                let outcome = outcomes.next().expect("no attempt left").map_err(Failure);
                async move { outcome }
            },
            || aborted,
        ));
        let after = groups.borrow().unkeyed();
        (used, result, after)
    }

    #[test]
    fn a_success_keeps_the_group() {
        let (used, result, after) = run(vec![Ok(())], false);
        assert!(result.is_ok());
        assert_eq!(used, [after]);
    }

    #[test]
    fn an_exit_failure_retries_once_on_a_fresh_token() {
        let (used, result, after) = run(vec![Err(ErrorKind::RemoteNetworkTimeout), Ok(())], false);
        assert!(result.is_ok());
        assert_eq!(used.len(), 2);
        assert_ne!(used[0], used[1]);
        assert_eq!(used[1], after);
    }

    #[test]
    fn a_failed_retry_moves_the_group_again_without_a_third_attempt() {
        let (used, result, after) = run(
            vec![
                Err(ErrorKind::RemoteStreamError),
                Err(ErrorKind::RemoteNetworkTimeout),
            ],
            false,
        );
        assert!(matches!(
            result,
            Err(ConnectError::Failed(Failure(
                ErrorKind::RemoteNetworkTimeout
            )))
        ));
        assert_eq!(used.len(), 2);
        assert!(!used.contains(&after));
    }

    #[test]
    fn a_refusal_is_returned_without_a_retry() {
        let (used, result, after) = run(vec![Err(ErrorKind::RemoteConnectionRefused)], false);
        assert!(matches!(result, Err(ConnectError::Failed(_))));
        assert_eq!(used, [after]);
    }

    #[test]
    fn a_failure_to_obtain_a_circuit_is_returned_without_a_retry() {
        let (used, result, after) = run(vec![Err(ErrorKind::TorNetworkTimeout)], false);
        assert!(matches!(result, Err(ConnectError::Failed(_))));
        assert_eq!(used, [after]);
    }

    #[test]
    fn an_abort_stops_the_retry_but_still_moves_the_group() {
        let (used, result, after) = run(vec![Err(ErrorKind::RemoteNetworkTimeout)], true);
        assert!(matches!(result, Err(ConnectError::Aborted)));
        assert_eq!(used.len(), 1);
        assert_ne!(used[0], after);
    }
}
