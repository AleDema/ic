use crate::{Computation, OpId, Operation};
use base64;
use ic_state_machine_tests::UserError;
use ic_state_machine_tests::WasmResult;
use ic_types::CanisterId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::{sync::RwLock, task::spawn_blocking, time};

// The maximum wait time for a computation to finish synchronously.
const DEFAULT_SYNC_WAIT_DURATION: Duration = Duration::from_millis(150);

pub const STATE_LABEL_HASH_SIZE: usize = 32;

/// Uniquely identifies a state.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct StateLabel(pub [u8; STATE_LABEL_HASH_SIZE]);

impl std::fmt::Debug for StateLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StateLabel(")?;
        self.0.iter().try_for_each(|b| write!(f, "{:02X}", b))?;
        write!(f, ")")
    }
}

impl std::convert::TryFrom<Vec<u8>> for StateLabel {
    // The input vector having the wrong size is the only possible error condition.
    type Error = InvalidSize;

    fn try_from(v: Vec<u8>) -> Result<StateLabel, InvalidSize> {
        if v.len() != STATE_LABEL_HASH_SIZE {
            return Err(InvalidSize);
        }

        let mut res = StateLabel::default();
        res.0[0..STATE_LABEL_HASH_SIZE].clone_from_slice(v.as_slice());
        Ok(res)
    }
}

// The only error condition is if the vector has the wrong size.
pub struct InvalidSize;

/// The state of the PocketIc-API.
///
/// The struct is Send + Sync and cloneable and can thus be shared between threads.
#[derive(Clone)]
pub struct PocketIcApiState<T> {
    // todo: this should become private at some point, pub for testing for now.
    inner: Arc<InnerApiState<T>>,
}

struct InnerApiState<T> {
    // impl note: If locks are acquired on both fields, acquire first on instances, then on graph.
    instances: RwLock<Vec<InstanceState<T>>>,
    graph: RwLock<HashMap<StateLabel, Computations>>,
    sync_wait_time: Duration,
}

pub struct PocketIcApiStateBuilder<T> {
    initial_instances: Vec<T>,
    sync_wait_time: Option<Duration>,
}

impl<T> PocketIcApiStateBuilder<T>
where
    T: HasStateLabel + Send,
{
    pub fn new() -> Self {
        Default::default()
    }

    /// Computations are dispatched into background tasks. If a computation takes longer than
    /// [sync_wait_time], the issue-operation returns, indicating that the given instance is busy.
    pub fn with_sync_wait_time(self, sync_wait_time: Duration) -> Self {
        Self {
            sync_wait_time: Some(sync_wait_time),
            ..self
        }
    }

    /// Will make the given instance available in the initial state.
    pub fn add_initial_instance(mut self, instance: T) -> Self {
        self.initial_instances.push(instance);
        self
    }

    pub fn build(self) -> PocketIcApiState<T> {
        let graph: HashMap<StateLabel, Computations> = self
            .initial_instances
            .iter()
            .map(|i| (i.get_state_label(), Computations::default()))
            .collect();
        let graph = RwLock::new(graph);

        let instances: Vec<_> = self
            .initial_instances
            .into_iter()
            .map(InstanceState::Available)
            .collect();
        let instances = RwLock::new(instances);

        let sync_wait_time = self.sync_wait_time.unwrap_or(DEFAULT_SYNC_WAIT_DURATION);
        let inner = Arc::new(InnerApiState {
            instances,
            graph,
            sync_wait_time,
        });
        PocketIcApiState { inner }
    }
}

impl<T> Default for PocketIcApiStateBuilder<T> {
    fn default() -> Self {
        Self {
            initial_instances: vec![],
            sync_wait_time: None,
        }
    }
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum OpOut {
    NoOutput,
    Time(u64),
    WasmResult(WasmResult),
    CanisterId(CanisterId),
    IcUserErr(UserError),
    Cycles(u128),
}

impl From<Result<WasmResult, UserError>> for OpOut {
    fn from(r: Result<WasmResult, UserError>) -> Self {
        match r {
            Ok(r) => Self::WasmResult(r),
            Err(e) => Self::IcUserErr(e),
        }
    }
}

impl From<Result<(), UserError>> for OpOut {
    fn from(r: Result<(), UserError>) -> Self {
        match r {
            Ok(_) => Self::NoOutput,
            Err(e) => Self::IcUserErr(e),
        }
    }
}

impl std::fmt::Debug for OpOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpOut::NoOutput => write!(f, "NoOutput"),
            OpOut::Time(x) => write!(f, "Time({})", x),
            OpOut::CanisterId(cid) => write!(f, "CanisterId({})", cid),
            OpOut::Cycles(x) => write!(f, "Cycles({})", x),
            OpOut::IcUserErr(x) => write!(f, "{}", x),
            OpOut::WasmResult(WasmResult::Reject(x)) => write!(f, "Reject({})", x),
            OpOut::WasmResult(WasmResult::Reply(bytes)) => {
                write!(f, "Reply({})", base64::encode(&bytes))
            }
        }
    }
}

pub type Computations = HashMap<OpId, (StateLabel, OpOut)>;

/// The PocketIcApiState has a vector with elements of InstanceState.
/// When an operation is bound to an instance, the corresponding element in the
/// vector is replaced by a Busy variant which contains information about the
/// computation that is currently running. Afterwards, the instance is put back as
/// Available.
#[derive(Clone)]
pub enum InstanceState<T> {
    Busy {
        state_label: StateLabel,
        op_id: OpId,
    },
    Available(T),
}

#[derive(Debug, Clone, Copy)]
pub enum IssueError {
    InstanceNotFound,
}

pub type IssueResult = std::result::Result<IssueOutcome, IssueError>;

/// An operation bound to an instance can be issued, that is, executed on the instance.
/// If the instance is already busy with an operation, the initial state and that operation
/// are returned.
/// If the result can be read from a cache, or if the computation is a fast read, an Output is
/// returned directly.
/// If the computation can be run and takes longer, a Busy variant is returned, containing the
/// requested op and the initial state.
/// TODO: The description implies three variants; two are currently represented as busy. We may
/// distinguish them with an additional HTTP response code, or maybe better as a third variant.
#[derive(Debug, PartialEq, Eq)]
pub enum IssueOutcome {
    /// The requested instance is busy executing this op on this state.
    Busy {
        state_label: StateLabel,
        op_id: OpId,
    },
    // This request is either cached or quickly executable, so we return
    // the output immediately.
    Output(OpOut),
}

impl IssueOutcome {
    pub fn get_busy(&self) -> Option<(StateLabel, OpId)> {
        match self {
            Self::Busy { state_label, op_id } => Some((state_label.clone(), op_id.clone())),
            _ => None,
        }
    }
}

/// This trait lets us put a mock of the pocket_ic into the PocketIcApiState.
pub trait HasStateLabel {
    fn get_state_label(&self) -> StateLabel;
}

impl<T> PocketIcApiState<T>
where
    T: HasStateLabel + Send + Sync + 'static,
{
    /// For polling:
    /// The client lib dispatches a long running operation and gets a Busy {state_label, op_id}.
    /// It then polls on that via this state tree api function.
    pub fn read_result(
        &self,
        state_label: &StateLabel,
        op_id: &OpId,
    ) -> Option<(StateLabel, OpOut)> {
        if let Some((new_state_label, op_out)) = self
            .inner
            .graph
            .try_read()
            .ok()?
            .get(state_label)?
            .get(op_id)
        {
            Some((new_state_label.clone(), op_out.clone()))
        } else {
            None
        }
    }

    /// An operation bound to an instance (a Computation) can be issued.
    ///
    /// * If the instance is busy executing an operation, the call returns [IssueOutcome::Busy]
    /// immediately. In that case, the state label and operation id contained in the result
    /// indicate that the instance is busy with a previous operation.
    ///
    /// * If the instance is available and the computation exceeds a (short) timeout,
    /// [IssueOutcome::Busy] is returned.
    ///
    /// * If the computation finished within the timeout, [IssueOutcome::Output] is returned
    /// containing the result.
    ///
    /// Operations are _not_ queued. Thus, if the instance is busy with an existing operation, the
    /// client has to retry until the operation is accepted.
    pub async fn issue<S>(&self, computation: Computation<S>) -> IssueResult
    where
        S: Operation<TargetType = T> + Send + 'static,
    {
        self.issue_with_timeout(computation, self.inner.sync_wait_time)
            .await
    }

    /// Same as [Self::issue] except that the timeout can be specified manually. This is useful in
    /// cases when clients want to enforce a long-running blocking call.
    pub async fn issue_with_timeout<S>(
        &self,
        computation: Computation<S>,
        sync_wait_time: Duration,
    ) -> IssueResult
    where
        S: Operation<TargetType = T> + Send + 'static,
    {
        let st = self.inner.clone();
        let mut instances = st.instances.write().await;
        let (bg_task, busy_outcome) =
            if let Some(instance_state) = instances.get(computation.instance_id) {
                // If this instance is busy, return the running op and initial state
                match instance_state {
                    // TODO: cache lookup possible with this state_label and our own op_id
                    InstanceState::Busy { state_label, op_id } => {
                        return Ok(IssueOutcome::Busy {
                            state_label: state_label.clone(),
                            op_id: op_id.clone(),
                        });
                    }
                    InstanceState::Available(mocket_ic) => {
                        let state_label = mocket_ic.get_state_label();
                        let op_id = computation.op.id();
                        let busy = InstanceState::Busy {
                            state_label: state_label.clone(),
                            op_id: op_id.clone(),
                        };
                        let InstanceState::Available(mut mocket_ic) =
                            std::mem::replace(&mut instances[computation.instance_id], busy)
                        else {
                            unreachable!()
                        };

                        let op = computation.op;
                        let instance_id = computation.instance_id;

                        // We schedule a blocking background task on the tokio runtime. Note that if all
                        // blocking workers are busy, the task is put on a queue (which is what we want).
                        //
                        // Note: One issue here is that we drop the join handle "on the floor". Threads
                        // that are not awaited upon before exiting the process are known to cause spurios
                        // issues. This should not be a problem as the tokio Executor will wait
                        // indefinitively for threads to return, unless a shutdown timeout is configured.
                        //
                        // See: https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html
                        let bg_task = spawn_blocking({
                            let op_id = op_id.clone();
                            let state_label = state_label.clone();
                            let st = self.inner.clone();
                            move || {
                                println!("Starting computation");
                                let result = op.compute(&mut mocket_ic);
                                let new_state_label = mocket_ic.get_state_label();
                                println!("Finished computation. Writing to graph.");
                                // add result to graph
                                let mut instances = st.instances.blocking_write();
                                let mut guard = st.graph.blocking_write();

                                let _ = std::mem::replace(
                                    &mut instances[instance_id],
                                    InstanceState::Available(mocket_ic),
                                );
                                let cached_computations =
                                    guard.entry(state_label.clone()).or_insert(HashMap::new());
                                cached_computations
                                    .insert(op_id.clone(), (new_state_label, result.clone()));
                                println!("Finished writing to graph");
                                result
                            }
                        });

                        // cache miss: replace pocket_ic instance in the vector with Busy
                        (bg_task, IssueOutcome::Busy { state_label, op_id })
                    }
                }
            } else {
                return Err(IssueError::InstanceNotFound);
            };
        // drop lock, otherwise we end up with a deadlock
        std::mem::drop(instances);

        // if the operation returns "in time", we return the result, otherwise we indicate to the
        // client that the instance is busy.
        //
        // note: this assumes that cancelling the JoinHandle does not stop the execution of the
        // background task. This only works because the background thread, in this case, is a
        // kernel thread.
        if let Ok(o) = time::timeout(sync_wait_time, bg_task).await {
            return Ok(IssueOutcome::Output(o.expect("join failed!")));
        }
        Ok(busy_outcome)
    }
}

impl<T: HasStateLabel> std::fmt::Debug for InstanceState<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy { state_label, op_id } => {
                write!(f, "Busy {{ {state_label:?}, {op_id:?} }}")?
            }
            Self::Available(pic) => write!(f, "Available({:?})", pic.get_state_label())?,
        }
        Ok(())
    }
}

impl<T: HasStateLabel> std::fmt::Debug for InnerApiState<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let instances = self.instances.blocking_read();
        let graph = self.graph.blocking_read();

        writeln!(f, "Instances:")?;
        for (idx, instance) in instances.iter().enumerate() {
            writeln!(f, "  [{idx}] {instance:?}")?;
        }

        writeln!(f, "Graph:")?;
        for (k, v) in graph.iter() {
            writeln!(f, "  {k:?} => {v:?}")?;
        }
        Ok(())
    }
}

impl<T: HasStateLabel> std::fmt::Debug for PocketIcApiState<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{:?}", self.inner)
    }
}
