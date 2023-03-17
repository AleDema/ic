#[rustfmt::skip]
mod unformatted {
//! <h1>Overview</h1>
//!
//! The *Artifact Manager* stores artifacts in the artifact pool. These
//! artifacts are used by the node it is running on and other nodes in the same
//! subnet. The *Artifact Manager* interacts with *Gossip* and its application
//! components:
//!
//!   * *Consensus*
//!   * *Distributed Key Generation*
//!   * *Certification*
//!   * *Ingress Manager*
//!   * *State Sync*
//!
//! It acts as a dispatcher for *Gossip* and ensures that the artifacts are
//! processed by the correct application components. It (de)multiplexes
//! artifacts to and from the different application components on behalf of
//! *Gossip* and bundles filters and priority functions.
//!
//! In order to let the *Consensus* components be stateless, the artifact
//! manager notifies the application components of artifacts received from
//! peers. The application components can then check if they are valid and
//! change their artifact pools (with a write lock to prevent conflicts and to
//! allow concurrent reads to the artifact pools).
//!
//! <h1>Properties</h1>
//!
//!   * All artifacts in the validated part of the artifact pool have been
//!     checked to be valid by the corresponding application component.
//!   * When new artifacts have been added to the artifact pool or when
//!     triggered by *Gossip*, the *Artifact Manager* asks the application
//!     components to check if they want to add new artifacts or move artifacts
//!     from the unvalidated part to the validated part of the pool.
//!   * When artifacts are added to the validated part of the artifact pool, the
//!     *Artifact Manager* notifies *Gossip* of adverts to send to peers.
//!     checked to be valid by the corresponding application component
//!   * When new artifacts have been added to the artifact pool or when
//!     triggered by Gossip the Artifact Manager asks the application components
//!     to check if they want to add new artifacts or move artifacts from the
//!     unvalidated part to the validated part of the pool
//!   * When artifacts are added to the validated part of the artifact pool, the
//!     Artifact Manager notifies Gossip of adverts to send to peers.
//!
//! <h1>High Level View</h1>
//!
//!
//!#                                                                 --------------------------
//!#                                                                 | ArtifactManagerBackend |
//!#                                                           |->   |     (Consensus)        |
//!#                                                           |     -------------------------
//!#                                                           |     --------------------------
//!#                                                           |     | ArtifactManagerBackend |
//!#                                                           |->   |       (Dkg)            |
//!#                                                           |     -------------------------
//!#     --------------          ------------------------      |     --------------------------
//!#     |   P2P      | <------> |  ArtifactManagerImpl |  ----|->   | ArtifactManagerBackend |
//!#     --------------          ------------------------      |     |     (Certification)    |
//!#                                                           |     --------------------------
//!#                                                           |     --------------------------
//!#                                                           |     | ArtifactManagerBackend |
//!#                                                           |->   |     (Ingress)          |
//!#                                                           |     -------------------------
//!#                                                           |     --------------------------
//!#                                                           |     | ArtifactManagerBackend |
//!#                                                           |->   |     (State Sync)       |
//!#                                                                 -------------------------
//!
//!  The main components are:
//!   * Front end
//!     manager::ArtifactManagerImpl implements the ArtifactManager trait and talks
//!     to P2P. It maintains the map of backends, one for each client: consensus, DKG,
//!     certification, ingress, state sync. It is just a light weight layer that routes the
//!     requests to the appropriate backend
//!
//!   * Back ends
//!     clients::ArtifactManagerBackend is a per-client wrapper that has two parts:
//!     1. Sync: Requests that can be served in the caller's context are processed by the
//!        sync part (e.g) has_artifact(), get_validated_by_identifier() that only need to
//!        look up the artifact pool
//!
//!        clients::ConsensusClient, etc implement the per-client sync part
//!
//!     2. Async: Processes the received artifacts via on_artifact(). The new artifacts are
//!        queued to a background worker thread. The thread runs a loop that calls into the
//!        per-client ArtifactProcessor implementation with the newly received artifacts
//!
//!        a. processors::ArtifactProcessorHandle manages the life cycle of these back ground
//!           threads, and queues the requests to the background thread via a crossbeam channel
//!        b. processors::ConsensusProcessor, etc implement the per-client ArtifactProcessor
//!           logic called by the threads. These roughly perform the sequence: add the new
//!           artifacts to the unvalidated pool, call the client.on_state_change(), apply the
//!           returned changes(mutations) to the artifact pools
//!
}

pub mod manager;
mod pool_readers;
mod processors;

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use ic_interfaces::{
    artifact_manager::{
        ArtifactClient, ArtifactPoolDescriptor, ArtifactProcessor, ProcessingResult,
    },
    artifact_pool::{ChangeSetProducer, MutablePool, UnvalidatedArtifact},
    canister_http::{CanisterHttpChangeSet, CanisterHttpPool},
    certification::ChangeSet as CertificationChangeSet,
    consensus_pool::ChangeSet as ConsensusChangeSet,
    dkg::ChangeSet as DkgChangeSet,
    ecdsa::{EcdsaChangeSet, EcdsaPool},
    gossip_pool::GossipPool,
    ingress_pool::{ChangeSet as IngressChangeSet, IngressPoolThrottler},
    time_source::{SysTimeSource, TimeSource},
};
use ic_logger::ReplicaLogger;
use ic_metrics::MetricsRegistry;
use ic_types::{artifact::*, artifact_kind::*, malicious_flags::MaliciousFlags, NodeId};
use prometheus::{histogram_opts, labels, Histogram};
use std::sync::{
    atomic::{AtomicBool, Ordering::SeqCst},
    Arc, RwLock,
};
use std::thread::{Builder as ThreadBuilder, JoinHandle};
use std::time::Duration;

/// Metrics for a client artifact processor.
struct ArtifactProcessorMetrics {
    /// The processing time histogram.
    processing_time: Histogram,
    /// The processing interval histogram.
    processing_interval: Histogram,
    /// The last update time.
    last_update: std::time::Instant,
}

impl ArtifactProcessorMetrics {
    /// The constructor creates a `ArtifactProcessorMetrics` instance.
    fn new(metrics_registry: MetricsRegistry, client: String) -> Self {
        let processing_time = metrics_registry.register(
            Histogram::with_opts(histogram_opts!(
                "artifact_manager_client_processing_time_seconds",
                "Artifact manager client processing time, in seconds",
                vec![
                    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.2, 2.5, 5.0, 8.0,
                    10.0, 15.0, 20.0, 50.0,
                ],
                labels! {"client".to_string() => client.clone()}
            ))
            .unwrap(),
        );
        let processing_interval = metrics_registry.register(
            Histogram::with_opts(histogram_opts!(
                "artifact_manager_client_processing_interval_seconds",
                "Duration between Artifact manager client processing, in seconds",
                vec![
                    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.2, 2.5, 5.0, 8.0,
                    10.0, 15.0, 20.0, 50.0,
                ],
                labels! {"client".to_string() => client}
            ))
            .unwrap(),
        );

        Self {
            processing_time,
            processing_interval,
            last_update: std::time::Instant::now(),
        }
    }

    fn with_metrics<T, F: FnOnce() -> T>(&mut self, run: F) -> T {
        self.processing_interval
            .observe((std::time::Instant::now() - self.last_update).as_secs_f64());
        let _timer = self.processing_time.start_timer();
        let result = run();
        self.last_update = std::time::Instant::now();
        result
    }
}

/// Manages the life cycle of the client specific artifact processor thread.
/// Also serves as the front end to enqueue requests to the processor thread.
pub struct ArtifactProcessorHandle<Artifact: ArtifactKind + 'static> {
    /// To send the process requests
    sender: Sender<UnvalidatedArtifact<Artifact::Message>>,
    /// Handle for the processing thread
    handle: Option<JoinHandle<()>>,
    /// To signal processing thread to exit.
    /// TODO: handle.abort() does not seem to work as expected
    shutdown: Arc<AtomicBool>,
}

impl<Artifact: ArtifactKind + 'static> ArtifactProcessorHandle<Artifact> {
    pub fn new<S: Fn(AdvertSendRequest<Artifact>) + Send + 'static>(
        time_source: Arc<SysTimeSource>,
        metrics_registry: MetricsRegistry,
        client: Box<dyn ArtifactProcessor<Artifact>>,
        send_advert: S,
    ) -> Self
    where
        <Artifact as ic_types::artifact::ArtifactKind>::Message: Send,
    {
        let (sender, receiver) = crossbeam_channel::unbounded();
        let shutdown = Arc::new(AtomicBool::new(false));

        // Spawn the processor thread
        let shutdown_cl = shutdown.clone();
        let handle = ThreadBuilder::new()
            .name(format!("{}_Processor", Artifact::TAG))
            .spawn(move || {
                Self::process_messages(
                    time_source,
                    client,
                    Box::new(send_advert),
                    receiver,
                    ArtifactProcessorMetrics::new(metrics_registry, Artifact::TAG.to_string()),
                    shutdown_cl,
                );
            })
            .unwrap();

        Self {
            sender,
            handle: Some(handle),
            shutdown,
        }
    }

    pub fn on_artifact(&self, artifact: UnvalidatedArtifact<Artifact::Message>) {
        self.sender
            .send(artifact)
            .unwrap_or_else(|err| panic!("Failed to send request: {:?}", err));
    }

    // The artifact processor thread loop
    #[allow(clippy::too_many_arguments)]
    fn process_messages<S: Fn(AdvertSendRequest<Artifact>) + Send + 'static>(
        time_source: Arc<SysTimeSource>,
        client: Box<dyn ArtifactProcessor<Artifact>>,
        send_advert: Box<S>,
        receiver: Receiver<UnvalidatedArtifact<Artifact::Message>>,
        mut metrics: ArtifactProcessorMetrics,
        shutdown: Arc<AtomicBool>,
    ) {
        let mut last_on_state_change_result = ProcessingResult::StateUnchanged;
        while !shutdown.load(SeqCst) {
            // TODO: assess impact of continued processing in same
            // iteration if StateChanged
            let recv_timeout = match last_on_state_change_result {
                ProcessingResult::StateChanged => Duration::from_millis(0),
                ProcessingResult::StateUnchanged => {
                    Duration::from_millis(ARTIFACT_MANAGER_TIMER_DURATION_MSEC)
                }
            };
            let recv_artifact = receiver.recv_timeout(recv_timeout);
            let batched_artifacts = match recv_artifact {
                Ok(artifact) => {
                    let mut artifacts = vec![artifact];
                    while let Ok(artifact) = receiver.try_recv() {
                        artifacts.push(artifact);
                    }
                    artifacts
                }
                Err(RecvTimeoutError::Timeout) => vec![],
                Err(RecvTimeoutError::Disconnected) => return,
            };
            time_source.update_time().ok();
            let (adverts, on_state_change_result) = metrics
                .with_metrics(|| client.process_changes(time_source.as_ref(), batched_artifacts));
            adverts.into_iter().for_each(&send_advert);
            last_on_state_change_result = on_state_change_result;
        }
    }
}

impl<Artifact: ArtifactKind + 'static> Drop for ArtifactProcessorHandle<Artifact> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.shutdown.store(true, SeqCst);
            handle.join().unwrap();
        }
    }
}

/// Periodic duration of `PollEvent` in milliseconds.
const ARTIFACT_MANAGER_TIMER_DURATION_MSEC: u64 = 200;

/// The struct contains all relevant interfaces for P2P to operate.
pub struct ArtifactClientHandle<Artifact: ArtifactKind + 'static> {
    /// Reference to the artifact client.
    pub pool_reader: Box<dyn ArtifactClient<Artifact>>,
    /// The artifact processor front end.
    pub processor_handle: ArtifactProcessorHandle<Artifact>,
    pub time_source: Arc<dyn TimeSource>,
}

pub fn create_ingress_handlers<
    PoolIngress: MutablePool<IngressArtifact, IngressChangeSet>
        + Send
        + Sync
        + GossipPool<IngressArtifact>
        + IngressPoolThrottler
        + 'static,
    S: Fn(AdvertSendRequest<IngressArtifact>) + Send + 'static,
>(
    send_advert: S,
    time_source: Arc<SysTimeSource>,
    ingress_pool: Arc<RwLock<PoolIngress>>,
    ingress_handler: Arc<
        dyn ChangeSetProducer<PoolIngress, ChangeSet = IngressChangeSet> + Send + Sync,
    >,
    log: ReplicaLogger,
    metrics_registry: MetricsRegistry,
    node_id: NodeId,
    malicious_flags: MaliciousFlags,
) -> ArtifactClientHandle<IngressArtifact> {
    let client = processors::IngressProcessor::new(ingress_pool.clone(), ingress_handler, node_id);
    let manager = ArtifactProcessorHandle::new(
        time_source.clone(),
        metrics_registry,
        Box::new(client),
        send_advert,
    );

    ArtifactClientHandle::<IngressArtifact> {
        processor_handle: manager,
        pool_reader: Box::new(pool_readers::IngressClient::new(
            time_source.clone(),
            ingress_pool,
            log,
            malicious_flags,
        )),
        time_source,
    }
}

pub fn create_consensus_handlers<
    PoolConsensus: MutablePool<ConsensusArtifact, ConsensusChangeSet>
        + Send
        + Sync
        + GossipPool<ConsensusArtifact>
        + 'static,
    C: ChangeSetProducer<PoolConsensus, ChangeSet = ConsensusChangeSet> + 'static,
    G: ArtifactPoolDescriptor<ConsensusArtifact, PoolConsensus> + 'static,
    S: Fn(AdvertSendRequest<ConsensusArtifact>) + Send + 'static,
>(
    send_advert: S,
    (consensus, consensus_gossip): (C, G),
    time_source: Arc<SysTimeSource>,
    consensus_pool: Arc<RwLock<PoolConsensus>>,
    log: ReplicaLogger,
    metrics_registry: MetricsRegistry,
) -> ArtifactClientHandle<ConsensusArtifact> {
    let client = processors::ConsensusProcessor::new(
        consensus_pool.clone(),
        Box::new(consensus),
        log,
        &metrics_registry,
    );
    let manager = ArtifactProcessorHandle::new(
        time_source.clone(),
        metrics_registry,
        Box::new(client),
        send_advert,
    );
    ArtifactClientHandle::<ConsensusArtifact> {
        processor_handle: manager,
        pool_reader: Box::new(pool_readers::ConsensusClient::new(
            consensus_pool,
            consensus_gossip,
        )),
        time_source,
    }
}

pub fn create_certification_handlers<
    PoolCertification: MutablePool<CertificationArtifact, CertificationChangeSet>
        + GossipPool<CertificationArtifact>
        + Send
        + Sync
        + 'static,
    C: ChangeSetProducer<PoolCertification, ChangeSet = CertificationChangeSet> + 'static,
    G: ArtifactPoolDescriptor<CertificationArtifact, PoolCertification> + 'static,
    S: Fn(AdvertSendRequest<CertificationArtifact>) + Send + 'static,
>(
    send_advert: S,
    (certifier, certifier_gossip): (C, G),
    time_source: Arc<SysTimeSource>,
    certification_pool: Arc<RwLock<PoolCertification>>,
    log: ReplicaLogger,
    metrics_registry: MetricsRegistry,
) -> ArtifactClientHandle<CertificationArtifact> {
    let client = processors::CertificationProcessor::new(
        certification_pool.clone(),
        Box::new(certifier),
        log,
        &metrics_registry,
    );
    let manager = ArtifactProcessorHandle::new(
        time_source.clone(),
        metrics_registry,
        Box::new(client),
        send_advert,
    );
    ArtifactClientHandle::<CertificationArtifact> {
        processor_handle: manager,
        pool_reader: Box::new(pool_readers::CertificationClient::new(
            certification_pool,
            certifier_gossip,
        )),
        time_source,
    }
}

pub fn create_dkg_handlers<
    PoolDkg: MutablePool<DkgArtifact, DkgChangeSet> + Send + Sync + GossipPool<DkgArtifact> + 'static,
    C: ChangeSetProducer<PoolDkg, ChangeSet = DkgChangeSet> + 'static,
    G: ArtifactPoolDescriptor<DkgArtifact, PoolDkg> + 'static,
    S: Fn(AdvertSendRequest<DkgArtifact>) + Send + 'static,
>(
    send_advert: S,
    (dkg, dkg_gossip): (C, G),
    time_source: Arc<SysTimeSource>,
    dkg_pool: Arc<RwLock<PoolDkg>>,
    log: ReplicaLogger,
    metrics_registry: MetricsRegistry,
) -> ArtifactClientHandle<DkgArtifact> {
    let client =
        processors::DkgProcessor::new(dkg_pool.clone(), Box::new(dkg), log, &metrics_registry);
    let manager = ArtifactProcessorHandle::new(
        time_source.clone(),
        metrics_registry,
        Box::new(client),
        send_advert,
    );
    ArtifactClientHandle::<DkgArtifact> {
        processor_handle: manager,
        pool_reader: Box::new(pool_readers::DkgClient::new(dkg_pool, dkg_gossip)),
        time_source,
    }
}

pub fn create_ecdsa_handlers<
    PoolEcdsa: MutablePool<EcdsaArtifact, EcdsaChangeSet>
        + Send
        + Sync
        + GossipPool<EcdsaArtifact>
        + EcdsaPool
        + 'static,
    C: ChangeSetProducer<PoolEcdsa, ChangeSet = EcdsaChangeSet> + 'static,
    G: ArtifactPoolDescriptor<EcdsaArtifact, PoolEcdsa> + 'static,
    S: Fn(AdvertSendRequest<EcdsaArtifact>) + Send + 'static,
>(
    send_advert: S,
    (ecdsa, ecdsa_gossip): (C, G),
    time_source: Arc<SysTimeSource>,
    ecdsa_pool: Arc<RwLock<PoolEcdsa>>,
    metrics_registry: MetricsRegistry,
    log: ReplicaLogger,
) -> ArtifactClientHandle<EcdsaArtifact> {
    let client = processors::EcdsaProcessor::new(
        ecdsa_pool.clone(),
        Box::new(ecdsa),
        log,
        &metrics_registry,
    );
    let manager = ArtifactProcessorHandle::new(
        time_source.clone(),
        metrics_registry,
        Box::new(client),
        send_advert,
    );
    ArtifactClientHandle::<EcdsaArtifact> {
        processor_handle: manager,
        pool_reader: Box::new(pool_readers::EcdsaClient::new(ecdsa_pool, ecdsa_gossip)),
        time_source,
    }
}

pub fn create_https_outcalls_handlers<
    PoolCanisterHttp: MutablePool<CanisterHttpArtifact, CanisterHttpChangeSet>
        + GossipPool<CanisterHttpArtifact>
        + Send
        + Sync
        + CanisterHttpPool
        + 'static,
    C: ChangeSetProducer<PoolCanisterHttp, ChangeSet = CanisterHttpChangeSet> + 'static,
    G: ArtifactPoolDescriptor<CanisterHttpArtifact, PoolCanisterHttp> + Send + Sync + 'static,
    S: Fn(AdvertSendRequest<CanisterHttpArtifact>) + Send + 'static,
>(
    send_advert: S,
    (pool_manager, canister_http_gossip): (C, G),
    time_source: Arc<SysTimeSource>,
    canister_http_pool: Arc<RwLock<PoolCanisterHttp>>,
    log: ReplicaLogger,
    metrics_registry: MetricsRegistry,
) -> ArtifactClientHandle<CanisterHttpArtifact> {
    let client = processors::CanisterHttpProcessor::new(
        canister_http_pool.clone(),
        Box::new(pool_manager),
        log,
    );
    let manager = ArtifactProcessorHandle::new(
        time_source.clone(),
        metrics_registry,
        Box::new(client),
        send_advert,
    );
    ArtifactClientHandle::<CanisterHttpArtifact> {
        processor_handle: manager,
        pool_reader: Box::new(pool_readers::CanisterHttpClient::new(
            canister_http_pool,
            canister_http_gossip,
        )),
        time_source,
    }
}
