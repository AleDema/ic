// A small Rust library that exports a setup() function to be used in system-tests,
// like the nns_upgrade_test, which sets up an IC with an NNS which is recovered
// from the latest mainnet backup.
//
// There are tests that use this library. Run them using either:
//
// * rm -rf test_tmpdir; ict testnet create recovered_mainnet_nns --lifetime-mins 120 --set-required-host-features=dc=zh1 --verbose -- --test_tmpdir=test_tmpdir
//
// * rm -rf test_tmpdir; ict test nns_upgrade_test --set-required-host-features=dc=zh1 -- --test_tmpdir=test_tmpdir --flaky_test_attempts=1

use candid::CandidType;
use canister_test::Canister;
use cycles_minting_canister::SetAuthorizedSubnetworkListArgs;
use dfn_candid::candid_one;
use flate2::read::GzDecoder;
use ic_canister_client::Sender;
use ic_canister_client_sender::{Ed25519KeyPair, SigKeys};
use ic_nervous_system_common::E8;
use ic_nns_common::types::NeuronId;
use ic_nns_governance::pb::v1::NnsFunction;
use ic_nns_test_utils::governance::submit_external_update_proposal;
use ic_recovery::nns_recovery_failover_nodes::{
    NNSRecoveryFailoverNodes, NNSRecoveryFailoverNodesArgs, StepType,
};
use ic_recovery::RecoveryArgs;
use ic_registry_subnet_type::SubnetType;
use ic_sns_wasm::pb::v1::{
    GetSnsSubnetIdsRequest, GetSnsSubnetIdsResponse, UpdateAllowedPrincipalsRequest,
    UpdateSnsSubnetListRequest,
};
use ic_tests::driver::boundary_node::BoundaryNodeVm;
use ic_tests::driver::constants::SSH_USERNAME;
use ic_tests::driver::driver_setup::SSH_AUTHORIZED_PRIV_KEYS_DIR;
use ic_tests::driver::universal_vm::DeployedUniversalVm;
use ic_tests::driver::universal_vm::UniversalVm;
use ic_tests::driver::{
    boundary_node::BoundaryNode,
    ic::{AmountOfMemoryKiB, ImageSizeGiB, InternetComputer, NrOfVCPUs, Subnet, VmResources},
    prometheus_vm::{HasPrometheus, PrometheusVm},
    test_env::{HasIcPrepDir, TestEnv, TestEnvAttribute},
    test_env_api::{
        retry, HasDependencies, HasIcDependencies, HasPublicApiUrl, HasTopologySnapshot,
        IcNodeContainer, IcNodeSnapshot, NnsCanisterWasmStrategy, NnsCustomizations, SshSession,
        TopologySnapshot,
    },
    universal_vm::UniversalVms,
};
use ic_tests::nns::{await_proposal_execution, get_canister};
use ic_tests::nns::{
    get_governance_canister, submit_update_elected_replica_versions_proposal,
    vote_execute_proposal_assert_executed,
};
use ic_tests::orchestrator::utils::rw_message::install_nns_with_customizations_and_check_progress;
use ic_tests::orchestrator::utils::subnet_recovery::set_sandbox_env_vars;
use ic_tests::util::{block_on, runtime_from_url};
use ic_types::{CanisterId, NodeId, PrincipalId, ReplicaVersion, SubnetId};
use icp_ledger::AccountIdentifier;
use serde::{Deserialize, Serialize};
use slog::{info, Logger};
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Cursor;
use std::io::Read;
use std::io::Write;
use std::net::IpAddr;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::str::FromStr;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::time::Duration;
use url::Url;

pub const OVERALL_TIMEOUT: Duration = Duration::from_secs(60 * 60);
pub const PER_TEST_TIMEOUT: Duration = Duration::from_secs(50 * 60);

// TODO: move this to an environment variable and set this on the CLI using --test_env=NNS_BACKUP_POD=zh1-pyr07.zh1.dfinity.network
const NNS_BACKUP_POD: &str = "zh1-pyr07.zh1.dfinity.network";
const NNS_BACKUP_POD_USER: &str = "dev";
const BOUNDARY_NODE_NAME: &str = "boundary-node-1";
const AUX_NODE_NAME: &str = "aux";
const RECOVERY_WORKING_DIR: &str = "recovery/working_dir";
const IC_CONFIG_DESTINATION: &str = "recovery/working_dir/ic.json5";
const NNS_STATE_DIR_PATH: &str = "recovery/working_dir/data";
const NNS_STATE_BACKUP_TARBALL_PATH: &str = "nns_state.tar.zst";
const IC_REPLAY: &str = "ic-replay";

/// Path to temporarily store a tarball of the IC registry local store.
/// This tarball will be created on the recovered NNS node and scp-ed from it.
const TMP_IC_REGISTRY_LOCAL_STORE_TARBALL_PATH: &str = "/tmp/ic_registry_local_store.tar.zst";

/// Path where to temporarily store the recovered NNS public key
/// before moving it to its final destination of /var/lib/ic/data/nns_public_key.pem.
const TMP_NNS_PUBLIC_KEY_PATH: &str = "/tmp/nns_public_key.pem";

const CONTROLLER: &str = "bc7vk-kulc6-vswcu-ysxhv-lsrxo-vkszu-zxku3-xhzmh-iac7m-lwewm-2ae";
const ORIGINAL_NNS_ID: &str = "tdb26-jop6k-aogll-7ltgs-eruif-6kk7m-qpktf-gdiqx-mxtrf-vb5e6-eqe";
const IC_CONFIG_SRC_PATH: &str = "/run/ic-node/config/ic.json5";
const SET_TESTNET_ENV_VARS_SH: &str = "set_testnet_env_variables.sh";
const RECOVERED_NNS: &str = "recovered-nns";
const MAINNET_GOVERNANCE_CANISTER_ID: &str = "rrkah-fqaaa-aaaaa-aaaaq-cai";
const MAINNET_SNS_WASM_CANISTER_ID: &str = "qaa6y-5yaaa-aaaaa-aaafa-cai";
const MAINNET_NNS_DAPP_CANISTER_ID: &str = "qoctq-giaaa-aaaaa-aaaea-cai";

#[derive(Deserialize, Serialize)]
pub struct RecoveredNnsNodeUrl {
    recovered_nns_node_url: Url,
}

impl TestEnvAttribute for RecoveredNnsNodeUrl {
    fn attribute_name() -> String {
        String::from("recovered_nns_node_url")
    }
}

#[derive(Deserialize, Serialize)]
pub struct RecoveredNnsDictatorNeuron {
    recovered_nns_dictator_neuron_id: NeuronId,
}

impl TestEnvAttribute for RecoveredNnsDictatorNeuron {
    fn attribute_name() -> String {
        String::from("recovered_nns_dictator_neuron_id")
    }
}

/// Sets up an IC running mainnet IC-OS nodes running the mainnet NNS
/// on the latest backup of the state of the mainnet NNS subnet.
///
/// The IC consists of a single-node system subnet and two unassigned nodes.
///
/// The mainnet NNS will be recovered to the first unassigned node.
///
/// The second unassigned node will be turned into an application subnet
/// to which we can deploy SNSes.
///
/// The single-node system subnet will run an initial NNS
/// that is required to perform the recovery but can be ignored after that.
pub fn setup(env: TestEnv) {
    // Fetch and unpack the NNS mainnet state backup concurrently with setting up the IC.
    // ic-replay also requires the ic.json5 config file of an NNS node.
    // Since we're creating the IC concurrently with fetching the state we use a channel to communicate
    // the IC topology to the thread fetching the backup such that the latter thread can later scp
    // the ic.json5 config file from the NNS node when it's online.
    let (tx_topology, rx_topology): (
        std::sync::mpsc::Sender<TopologySnapshot>,
        Receiver<TopologySnapshot>,
    ) = mpsc::channel();
    let (tx_aux_node, rx_aux_node): (
        std::sync::mpsc::Sender<DeployedUniversalVm>,
        Receiver<DeployedUniversalVm>,
    ) = mpsc::channel();

    // Recover the NNS concurrently:
    let env_clone = env.clone();
    let nns_state_thread = std::thread::spawn(move || {
        setup_recovered_nns(env_clone, rx_topology, rx_aux_node);
    });

    // Start a p8s VM concurrently:
    let env_clone = env.clone();
    let prometheus_thread = std::thread::spawn(move || {
        PrometheusVm::default()
            .with_vm_resources(VmResources {
                vcpus: Some(NrOfVCPUs::new(32)),
                memory_kibibytes: Some(AmountOfMemoryKiB::new(125000000)), // ~128 GiB
                boot_image_minimal_size_gibibytes: Some(ImageSizeGiB::new(500)),
            })
            .start(&env_clone)
            .expect("Failed to start prometheus VM");
    });

    // Setup and start the aux UVM concurrently:
    let env_clone = env.clone();
    let uvm_thread = std::thread::spawn(move || {
        UniversalVm::new(AUX_NODE_NAME.to_string())
            .start(&env_clone)
            .expect("Failed to start Universal VM");
    });

    setup_ic(env.clone());

    let env_clone = env.clone();
    std::thread::spawn(move || {
        // Send the IC topology to the thread fetching the nns state (nns_state_thread)
        // such that it can scp the ic.json5 config file required by ic-replay.
        let topology = env_clone.topology_snapshot();
        tx_topology.send(topology).unwrap();

        uvm_thread.join().unwrap();
        let deployed_universal_vm = env_clone.get_deployed_universal_vm(AUX_NODE_NAME).unwrap();
        tx_aux_node.send(deployed_universal_vm).unwrap();
    });

    nns_state_thread
        .join()
        .unwrap_or_else(|e| std::panic::resume_unwind(e));

    prometheus_thread.join().unwrap();
    env.sync_with_prometheus_by_name(RECOVERED_NNS);
}

fn setup_recovered_nns(
    env: TestEnv,
    rx_topology: Receiver<TopologySnapshot>,
    rx_aux_node: Receiver<DeployedUniversalVm>,
) {
    let logger = env.logger();

    let env_clone = env.clone();
    let fetch_mainnet_ic_replay_thread = std::thread::spawn(move || {
        fetch_mainnet_ic_replay(env_clone);
    });

    fetch_nns_state_from_backup_pod(env.clone());

    let topology = rx_topology.recv().unwrap();
    let nns_node = topology.root_subnet().nodes().next().unwrap();
    let mut unassigned_nodes = topology.unassigned_nodes();
    let recovered_nns_node = unassigned_nodes.next().unwrap();
    let new_subnet_node = unassigned_nodes.next().unwrap();
    fetch_ic_config(env.clone(), nns_node.clone());

    let principal = dfx_import_identity(env.clone());

    let account_id = AccountIdentifier::new(principal, None);
    info!(logger, "account_id = {account_id}");

    // The following ensures ic-replay and ic-recovery know where to get their required dependencies.
    let recovery_dir = env.get_dependency_path("rs/tests");
    set_sandbox_env_vars(recovery_dir.join("recovery/binaries"));

    fetch_mainnet_ic_replay_thread
        .join()
        .unwrap_or_else(|e| panic!("Failed to fetch the mainnet ic-replay because {e:?}"));

    let neuron_id: NeuronId = prepare_nns_state(env.clone(), account_id);

    let aux_node = rx_aux_node.recv().unwrap();

    recover_nns_subnet(env.clone(), nns_node, recovered_nns_node.clone(), aux_node);
    test_recovered_nns(env.clone(), neuron_id, recovered_nns_node.clone());

    let recovered_nns_public_key =
        save_recovered_nns_public_key(env.clone(), recovered_nns_node.clone());

    let env_clone = env.clone();
    let recovered_nns_node_clone = recovered_nns_node.clone();
    let recovered_nns_public_key_clone = recovered_nns_public_key.clone();
    let bn_thread = std::thread::spawn(move || {
        setup_boundary_node(
            env_clone,
            recovered_nns_node_clone,
            recovered_nns_public_key_clone,
        )
    });

    let wallet_canister_id = support_snses(
        env.clone(),
        recovered_nns_node.clone(),
        new_subnet_node.clone(),
        recovered_nns_public_key.clone(),
        neuron_id,
        principal,
    );

    let boundary_node_url = bn_thread.join().unwrap();

    write_sh_lib(
        env,
        neuron_id,
        new_subnet_node,
        wallet_canister_id,
        boundary_node_url,
    );
}

/// Configure the testnet in such a way that it supports creating SNSes. Specifically:
///
/// * Create an application subnet from the 2nd unassigned node.
/// * Configure the NNS to allow creating SNSes on that application subnet.
/// * Create a cycles wallet.
/// * Allow the cycles wallet to create SNSes.
fn support_snses(
    env: TestEnv,
    recovered_nns_node: IcNodeSnapshot,
    new_subnet_node: IcNodeSnapshot,
    recovered_nns_public_key: PathBuf,
    neuron_id: NeuronId,
    principal: PrincipalId,
) -> CanisterId {
    create_subnet(
        env.clone(),
        recovered_nns_node.clone(),
        neuron_id,
        new_subnet_node.clone(),
    );

    move_node_to_recovered_nns(
        env.clone(),
        recovered_nns_node.clone(),
        new_subnet_node.clone(),
        recovered_nns_public_key.clone(),
    );

    wait_until_ready_for_interaction(env.logger(), new_subnet_node.clone());

    let new_subnet_id = get_app_subnet_id(
        env.clone(),
        recovered_nns_node.clone(),
        new_subnet_node.node_id,
    );

    set_default_subnets(
        env.clone(),
        recovered_nns_node.clone(),
        neuron_id,
        new_subnet_id,
    );

    let wallet_canister_id = create_cycles_wallet(
        env.clone(),
        recovered_nns_node.clone(),
        principal,
        new_subnet_node.get_public_url(),
    );

    configure_sns_wasms(
        env.clone(),
        neuron_id,
        recovered_nns_node.clone(),
        new_subnet_id,
        wallet_canister_id,
    );
    wallet_canister_id
}

/// Start an IC and install the initial NNS.
fn setup_ic(env: TestEnv) {
    InternetComputer::new()
        .with_default_vm_resources(VmResources {
            vcpus: Some(NrOfVCPUs::new(16)),
            memory_kibibytes: None,
            boot_image_minimal_size_gibibytes: Some(ImageSizeGiB::new(500)),
        })
        .add_subnet(Subnet::new(SubnetType::System).add_nodes(1))
        .with_unassigned_nodes(2)
        .with_mainnet_config()
        .setup_and_start(&env)
        .expect("Failed to setup IC under test");
    install_nns_with_customizations_and_check_progress(
        env.topology_snapshot(),
        NnsCanisterWasmStrategy::TakeBuiltFromSources,
        NnsCustomizations::default(),
    );
}

fn save_recovered_nns_public_key(env: TestEnv, recovered_nns_node: IcNodeSnapshot) -> PathBuf {
    let ic_admin_path = env
        .clone()
        .get_dependency_path("rs/tests/recovery/binaries/ic-admin");
    let recovered_nns_url = recovered_nns_node.get_public_url();
    let recovered_nns_public_key = env.clone().get_path("recovered_nns_pubkey.pem");
    let mut cmd = Command::new(ic_admin_path);
    cmd.arg("--nns-url")
        .arg(recovered_nns_url.to_string())
        .arg("get-subnet-public-key")
        .arg(ORIGINAL_NNS_ID)
        .arg(recovered_nns_public_key.clone());
    info!(env.logger(), "{cmd:?} ...");
    cmd.output().unwrap_or_else(|e| {
        panic!("Could not get the public key of the recovered NNS because {e:?}",)
    });
    recovered_nns_public_key
}

fn setup_boundary_node(
    env: TestEnv,
    recovered_nns_node: IcNodeSnapshot,
    recovered_nns_public_key: PathBuf,
) -> Url {
    let logger = env.logger();
    let recovered_nns_url = recovered_nns_node.get_public_url();
    BoundaryNode::new(String::from(BOUNDARY_NODE_NAME))
        .allocate_vm(&env)
        .expect("Allocation of BoundaryNode failed.")
        .for_ic(&env, "")
        .with_nns_public_key(recovered_nns_public_key)
        .with_nns_urls(vec![recovered_nns_url])
        .use_real_certs_and_dns()
        .start(&env)
        .expect("failed to setup BoundaryNode VM");

    let boundary_node = env
        .clone()
        .get_deployed_boundary_node(BOUNDARY_NODE_NAME)
        .unwrap()
        .get_snapshot()
        .unwrap();

    info!(
        logger,
        "Waiting until the ic-registry-replicator on {BOUNDARY_NODE_NAME} has finished local store initialization ..."
    );
    retry(
        logger.clone(),
        Duration::from_secs(500),
        Duration::from_secs(5),
        || {
            boundary_node.block_on_bash_script(
                r#"
                    set -e
                    until journalctl -u ic-registry-replicator -g "Finished local store initialization"; do
                        sleep 1
                    done
                    sudo systemctl restart control-plane.service
                "#
            )
        },
    )
    .unwrap_or_else(|e| {
        panic!("The ic-registry-replicator on {BOUNDARY_NODE_NAME} didn't finish local store initialization in time. Error: {e:?}")
    });

    let recovered_nns_node_id = recovered_nns_node.node_id;
    boundary_node.block_on_bash_script(&format!(r#"
        set -e
        cp /etc/nginx/conf.d/002-mainnet-nginx.conf /tmp/
        sed 's/set $subnet_id "$random_route_subnet_id";/set $subnet_id "{ORIGINAL_NNS_ID}";/' -i /tmp/002-mainnet-nginx.conf
        sed 's/set $subnet_type "$random_route_subnet_type";/set $subnet_type "system";/' -i /tmp/002-mainnet-nginx.conf
        sed 's/set $node_id "$random_route_node_id";/set $node_id "{recovered_nns_node_id}";/' -i /tmp/002-mainnet-nginx.conf
        sudo mount --bind /tmp/002-mainnet-nginx.conf /etc/nginx/conf.d/002-mainnet-nginx.conf
        sudo systemctl reload nginx
    "#)).unwrap_or_else(|e| {
        panic!("Could not reconfigure nginx on {BOUNDARY_NODE_NAME} to only route to the recovered NNS because {e:?}",)
    });

    info!(logger, "Waiting until {BOUNDARY_NODE_NAME} is healthy ...");
    boundary_node
        .await_status_is_healthy()
        .expect("Boundary Node {BOUNDARY_NODE_NAME} did not come up!");

    let playnet = boundary_node.playnet.unwrap();
    info!(
        logger,
        "NNS Dapp: https://{MAINNET_NNS_DAPP_CANISTER_ID}.{playnet}"
    );
    Url::from_str(&format!("https://{playnet}")).unwrap()
}

fn fetch_nns_state_from_backup_pod(env: TestEnv) {
    let target = format!("{NNS_BACKUP_POD_USER}@{NNS_BACKUP_POD}:/home/{NNS_BACKUP_POD_USER}/{NNS_STATE_BACKUP_TARBALL_PATH}");
    let logger: slog::Logger = env.logger();
    let nns_state_backup_path = env.get_path(NNS_STATE_BACKUP_TARBALL_PATH);
    info!(
        logger,
        "Downloading {} to {:?} ...",
        target,
        nns_state_backup_path.clone()
    );
    // TODO: consider using the ssh2 crate (like we do in prometheus_vm.rs)
    // instead of shelling out to scp.
    let mut cmd = Command::new("scp");
    cmd.arg("-oUserKnownHostsFile=/dev/null")
        .arg("-oStrictHostKeyChecking=no")
        .arg(target.clone())
        .arg(nns_state_backup_path.clone());
    info!(env.logger(), "{cmd:?} ...");
    let scp_out = cmd.output().unwrap_or_else(|e| {
        panic!(
            "Could not scp the {NNS_STATE_BACKUP_TARBALL_PATH} from the backup pod because: {e:?}!",
        )
    });
    if !scp_out.status.success() {
        std::io::stdout().write_all(&scp_out.stdout).unwrap();
        std::io::stderr().write_all(&scp_out.stderr).unwrap();
        panic!("Could not scp the {NNS_STATE_BACKUP_TARBALL_PATH} from the backup pod!");
    }
    info!(
        logger,
        "Downloaded {target:} to {:?}, unpacking ...", nns_state_backup_path
    );
    let mut cmd = Command::new("tar");
    cmd.arg("xf")
        .arg(nns_state_backup_path.clone())
        .arg("-C")
        .arg(env.base_path())
        .arg(format!("--transform=s|nns_state/|{NNS_STATE_DIR_PATH}/|"));
    info!(env.logger(), "{cmd:?} ...");
    let tar_out = cmd
        .output()
        .expect("Could not unpack {NNS_STATE_BACKUP_TARBALL_PATH}!");
    if !tar_out.status.success() {
        std::io::stdout().write_all(&tar_out.stdout).unwrap();
        std::io::stderr().write_all(&tar_out.stderr).unwrap();
        panic!("Could not unpack {NNS_STATE_BACKUP_TARBALL_PATH}!");
    }
    info!(logger, "Unpacked {:?}", nns_state_backup_path);
}

fn fetch_ic_config(env: TestEnv, nns_node: IcNodeSnapshot) {
    let logger: slog::Logger = env.logger();
    let nns_node_ip = nns_node.get_ip_addr();
    info!(
        logger,
        "Setting up SSH session to NNS node with IP {nns_node_ip:?} ..."
    );
    let session = nns_node.block_on_ssh_session().unwrap_or_else(|e| {
        panic!("Failed to setup SSH session to NNS node with IP {nns_node_ip:?} because: {e:?}!",)
    });

    let destination_dir = env.get_path(RECOVERY_WORKING_DIR);
    std::fs::create_dir_all(destination_dir.clone()).unwrap_or_else(|e| {
        panic!("Couldn't create directory {destination_dir:?} because {e}!");
    });
    let destination = env.get_path(IC_CONFIG_DESTINATION);
    info!(
        logger,
        "scp-ing {nns_node_ip:?}:{IC_CONFIG_SRC_PATH:} to {destination:?} ..."
    );
    // scp the ic.json5 of the NNS node to the nns_state directory in the local test environment.
    let (mut remote_ic_config_file, _) = session
        .scp_recv(Path::new(IC_CONFIG_SRC_PATH))
        .unwrap_or_else(|e| {
            panic!("Failed to scp {nns_node_ip:?}:{IC_CONFIG_SRC_PATH:} because: {e:?}!",)
        });
    let mut destination_file = File::create(&destination)
        .unwrap_or_else(|e| panic!("Failed to open destination {destination:?} because: {e:?}"));
    std::io::copy(&mut remote_ic_config_file, &mut destination_file).unwrap_or_else(|e| {
        panic!(
            "Failed to scp {nns_node_ip:?}:{IC_CONFIG_SRC_PATH:} to {destination:?} because {e:?}!"
        )
    });
    info!(
        logger,
        "Successfully scp-ed {nns_node_ip:?}:{IC_CONFIG_SRC_PATH:} to {destination:?}."
    );
}

fn ic_replay(env: TestEnv, mut mutate_cmd: impl FnMut(&mut Command)) -> Output {
    let logger: slog::Logger = env.logger();
    let ic_replay_path = env.get_path(IC_REPLAY);
    let subnet_id = SubnetId::from(PrincipalId::from_str(ORIGINAL_NNS_ID).unwrap());
    let nns_state_dir = env.get_path(NNS_STATE_DIR_PATH);
    let ic_config_file = env.get_path(IC_CONFIG_DESTINATION);

    let mut cmd = Command::new(ic_replay_path);
    cmd.arg("--subnet-id")
        .arg(subnet_id.to_string())
        .arg("--data-root")
        .arg(nns_state_dir.clone())
        .arg(ic_config_file.clone());
    mutate_cmd(&mut cmd);
    info!(logger, "{cmd:?} ...");
    let ic_replay_out = cmd.output().expect("Failed to run {cmd:?}");
    if !ic_replay_out.status.success() {
        std::io::stdout().write_all(&ic_replay_out.stdout).unwrap();
        std::io::stderr().write_all(&ic_replay_out.stderr).unwrap();
        panic!("Failed to run {cmd:?}!");
    }
    ic_replay_out
}

fn with_neuron_for_tests(env: TestEnv) -> NeuronId {
    let logger: slog::Logger = env.logger();
    let controller = PrincipalId::from_str(CONTROLLER).unwrap();

    info!(logger, "Create a neuron followed by trusted neurons ...");
    let neuron_stake_e8s: u64 = 1_000_000_000 * E8;
    let ic_replay_out = ic_replay(env, |cmd| {
        cmd.arg("with-neuron-for-tests")
            .arg(controller.to_string())
            .arg(neuron_stake_e8s.to_string());
    });

    let prefix = "neuron_id=";
    let neuron_id = match std::str::from_utf8(&ic_replay_out.stdout)
        .unwrap()
        .split('\n')
        .filter(|line| line.starts_with(prefix))
        .collect::<Vec<&str>>()
        .first()
        .unwrap()
        .split(prefix)
        .collect::<Vec<&str>>()[..]
    {
        [_, neuron_id_str] => NeuronId(neuron_id_str.parse::<u64>().unwrap()),
        _ => panic!("Line didn't start with \"neuron_id=\"!"),
    };
    info!(logger, "Created neuron with id {neuron_id:?}");
    neuron_id
}

fn with_trusted_neurons_following_neuron_for_tests(env: TestEnv, neuron_id: NeuronId) {
    let NeuronId(id) = neuron_id;
    let controller = PrincipalId::from_str(CONTROLLER).unwrap();
    ic_replay(env, |cmd| {
        cmd.arg("with-trusted-neurons-following-neuron-for-tests")
            .arg(id.to_string())
            .arg(controller.to_string());
    });
}

fn with_ledger_account_for_tests(env: TestEnv, account_id: AccountIdentifier) {
    let logger: slog::Logger = env.logger();
    info!(logger, "Giving our principal 1 million ICP ...");
    ic_replay(env, |cmd| {
        cmd.arg("with-ledger-account-for-tests")
            .arg(account_id.to_string())
            .arg((1_000_000 * E8).to_string());
    });
    info!(logger, "Our principal now has 1 million ICP");
}

fn fetch_mainnet_ic_replay(env: TestEnv) {
    let logger = env.logger();
    let version = env
        .read_dependency_to_string("testnet/mainnet_nns_revision.txt")
        .unwrap();
    let mainnet_ic_replica_url =
        format!("https://download.dfinity.systems/ic/{version}/release/ic-replay.gz");
    let ic_replay_path = env.get_path(IC_REPLAY);
    let ic_replay_gz_path = env.get_path("ic-replay.gz");
    // let mut tmp_file = tempfile::tempfile().unwrap();
    info!(
        logger,
        "Downloading {mainnet_ic_replica_url:?} to {ic_replay_gz_path:?} ..."
    );
    let response = reqwest::blocking::get(mainnet_ic_replica_url.clone())
        .unwrap_or_else(|e| panic!("Failed to download {mainnet_ic_replica_url:?} because {e:?}"));
    if !response.status().is_success() {
        panic!("Failed to download {mainnet_ic_replica_url}");
    }
    let bytes = response.bytes().unwrap();
    let mut content = Cursor::new(bytes);
    let mut ic_replay_gz_file = File::create(ic_replay_gz_path.clone()).unwrap();
    std::io::copy(&mut content, &mut ic_replay_gz_file).unwrap_or_else(|e| {
        panic!("Can't copy {mainnet_ic_replica_url} to {ic_replay_gz_path:?} because {e:?}")
    });
    info!(
        logger,
        "Downloaded {mainnet_ic_replica_url:?} to {ic_replay_gz_path:?}. Uncompressing to {ic_replay_path:?} ..."
    );
    let ic_replay_gz_file = File::open(ic_replay_gz_path.clone()).unwrap();
    let mut gz = GzDecoder::new(&ic_replay_gz_file);
    let mut ic_replay_file = OpenOptions::new()
        .create(true)
        .write(true)
        .mode(0o755)
        .open(ic_replay_path.clone())
        .unwrap();
    std::io::copy(&mut gz, &mut ic_replay_file).unwrap_or_else(|e| {
        panic!("Can't uncompress {ic_replay_gz_path:?} to {ic_replay_path:?} because {e:?}")
    });
    info!(
        logger,
        "Uncompressed {ic_replay_gz_path:?} to {ic_replay_path:?}"
    );
}

fn prepare_nns_state(env: TestEnv, account_id: AccountIdentifier) -> NeuronId {
    let neuron_id = with_neuron_for_tests(env.clone());
    with_trusted_neurons_following_neuron_for_tests(env.clone(), neuron_id);
    with_ledger_account_for_tests(env.clone(), account_id);
    neuron_id
}

fn recover_nns_subnet(
    env: TestEnv,
    nns_node: IcNodeSnapshot,
    recovered_nns_node: IcNodeSnapshot,
    aux_node: DeployedUniversalVm,
) {
    let logger = env.logger();

    info!(
        logger,
        "Waiting until the {AUX_NODE_NAME} node is reachable over SSH before we run ic-recovery ..."
    );
    let _session = aux_node.block_on_ssh_session();

    info!(logger, "Starting ic-recovery ...");
    let recovery_binaries_path =
        std::fs::canonicalize(env.get_dependency_path("rs/tests/recovery/binaries")).unwrap();

    let dir = env.base_path();
    std::os::unix::fs::symlink(recovery_binaries_path, dir.join("recovery/binaries")).unwrap();

    let nns_url: url::Url = nns_node.get_public_url();
    let replica_version = env.get_initial_replica_version().unwrap();
    let subnet_id = SubnetId::from(PrincipalId::from_str(ORIGINAL_NNS_ID).unwrap());
    let aux_ip = aux_node.get_vm().unwrap().ipv6;
    let priv_key_path = env
        .get_path(SSH_AUTHORIZED_PRIV_KEYS_DIR)
        .join(SSH_USERNAME);
    let nns_ip = nns_node.get_ip_addr();
    let upload_ip = recovered_nns_node.get_ip_addr();

    let recovery_args = RecoveryArgs {
        dir,
        nns_url: nns_url.clone(),
        replica_version: Some(replica_version.clone()),
        key_file: Some(priv_key_path),
        test_mode: true,
    };

    let nns_recovery_failover_nodes_args = NNSRecoveryFailoverNodesArgs {
        subnet_id,
        replica_version: Some(replica_version),
        aux_ip: Some(IpAddr::V6(aux_ip)),
        aux_user: Some(SSH_USERNAME.to_string()),
        registry_url: None,
        validate_nns_url: nns_url,
        download_node: None,
        upload_node: Some(upload_ip),
        parent_nns_host_ip: Some(nns_ip),
        replacement_nodes: Some(vec![recovered_nns_node.node_id]),
        next_step: None,
    };

    let nns_recovery_failover_nodes = NNSRecoveryFailoverNodes::new(
        logger.clone(),
        recovery_args,
        None,
        nns_recovery_failover_nodes_args,
        false,
    );

    // go over all steps of the NNS recovery
    for (step_type, step) in nns_recovery_failover_nodes {
        if step_type == StepType::DownloadCertifications
            || step_type == StepType::MergeCertificationPools
            || step_type == StepType::ValidateReplayOutput
        {
            info!(logger, "Skipping step: {:?}", step_type);
            continue;
        }
        info!(logger, "Executing step: {:?}", step_type);
        info!(logger, "{}", step.descr());
        step.exec()
            .unwrap_or_else(|e| panic!("Execution of step {:?} failed: {}", step_type, e));
    }

    wait_until_ready_for_interaction(logger.clone(), recovered_nns_node);
}

fn wait_until_ready_for_interaction(logger: Logger, node: IcNodeSnapshot) {
    let node_ip = node.get_ip_addr();
    info!(
        logger.clone(),
        "Waiting until node {node_ip:?} is ready for interaction ..."
    );
    retry(
        logger.clone(),
        Duration::from_secs(500),
        Duration::from_secs(5),
        || node.block_on_bash_script("journalctl | grep -q 'Ready for interaction'"),
    )
    .unwrap_or_else(|e| {
        panic!("Node {node_ip:?} didn't become ready for interaction in time because {e:?}")
    });

    info!(logger, "Node {node_ip:?} is ready for interaction.");
}

fn test_recovered_nns(env: TestEnv, neuron_id: NeuronId, nns_node: IcNodeSnapshot) {
    let logger: slog::Logger = env.clone().logger();
    info!(logger, "Testing recovered NNS ...");
    let contents = env
        .clone()
        .read_dependency_to_string("rs/tests/nns/secret_key.pem")
        .expect("Could not read rs/tests/nns/secret_key.pem");
    let sig_keys =
        SigKeys::from_pem(&contents).expect("Failed to parse rs/tests/nns/secret_key.pem");
    let proposal_sender = Sender::SigKeys(sig_keys);
    bless_replica_version(
        env.clone(),
        neuron_id,
        proposal_sender,
        nns_node.clone(),
        "1111111111111111111111111111111111111111".to_string(),
    );
    let recovered_nns_node_url = nns_node.get_public_url();
    RecoveredNnsNodeUrl {
        recovered_nns_node_url: recovered_nns_node_url.clone(),
    }
    .write_attribute(&env);
    RecoveredNnsDictatorNeuron {
        recovered_nns_dictator_neuron_id: neuron_id,
    }
    .write_attribute(&env);
    info!(
        logger,
        "Successfully recovered NNS at {}. Interact with it using {:?}.",
        recovered_nns_node_url.clone(),
        neuron_id,
    );
}

fn package_registry_local_store(logger: Logger, recovered_nns_node: IcNodeSnapshot) {
    retry(
        logger,
        Duration::from_secs(120),
        Duration::from_secs(5),
        || {
            recovered_nns_node.block_on_bash_script(&format!(
                r#"
                    sudo tar -C /var/lib/ic/data \
                        -cf {TMP_IC_REGISTRY_LOCAL_STORE_TARBALL_PATH} \
                        --use-compress-program='zstd --threads=0' \
                        ic_registry_local_store
                "#
            ))
        },
    )
    .unwrap_or_else(|e| panic!("Could not create ic_registry_local_store.tar.zst because {e:?}",));
}

fn move_node_to_recovered_nns(
    env: TestEnv,
    recovered_nns_node: IcNodeSnapshot,
    new_subnet_node: IcNodeSnapshot,
    recovered_nns_public_key: PathBuf,
) {
    let logger = env.logger();
    let new_subnet_node_ip = new_subnet_node.get_ip_addr();
    info!(
        logger,
        "Moving node {new_subnet_node_ip:?} to the recovered NNS ..."
    );
    package_registry_local_store(logger.clone(), recovered_nns_node.clone());

    let priv_key_path = env
        .get_path(SSH_AUTHORIZED_PRIV_KEYS_DIR)
        .join(SSH_USERNAME);
    let recovered_nns_node_ip = recovered_nns_node.get_ip_addr();
    let src = format!(
        "{SSH_USERNAME}@[{recovered_nns_node_ip:?}]:{TMP_IC_REGISTRY_LOCAL_STORE_TARBALL_PATH}"
    );
    let dst = format!(
        "{SSH_USERNAME}@[{new_subnet_node_ip:?}]:{TMP_IC_REGISTRY_LOCAL_STORE_TARBALL_PATH}"
    );

    let mut cmd = Command::new("scp");
    cmd.arg("-oUserKnownHostsFile=/dev/null")
        .arg("-oStrictHostKeyChecking=no")
        .arg("-3")
        .arg("-i")
        .arg(priv_key_path.clone())
        .arg(src.clone())
        .arg(dst.clone());
    info!(logger, "{cmd:?} ...");
    let scp_out = cmd
        .output()
        .unwrap_or_else(|e| panic!("Could not scp {src} to {dst} because: {e:?}!",));
    if !scp_out.status.success() {
        std::io::stdout().write_all(&scp_out.stdout).unwrap();
        std::io::stderr().write_all(&scp_out.stderr).unwrap();
        panic!("Could not scp {src} to {dst}!");
    }

    let dst_nns_public_key_pem =
        format!("{SSH_USERNAME}@[{new_subnet_node_ip:?}]:{TMP_NNS_PUBLIC_KEY_PATH}");
    let mut cmd = Command::new("scp");
    cmd.arg("-oUserKnownHostsFile=/dev/null")
        .arg("-oStrictHostKeyChecking=no")
        .arg("-i")
        .arg(priv_key_path)
        .arg(recovered_nns_public_key.clone())
        .arg(dst_nns_public_key_pem.clone());
    info!(logger, "{cmd:?} ...");
    let scp_out = cmd
        .output()
        .unwrap_or_else(|e| panic!("Could not scp {recovered_nns_public_key:?} to {dst_nns_public_key_pem} because: {e:?}!",));
    if !scp_out.status.success() {
        std::io::stdout().write_all(&scp_out.stdout).unwrap();
        std::io::stderr().write_all(&scp_out.stderr).unwrap();
        panic!("Could not scp {recovered_nns_public_key:?} to {dst_nns_public_key_pem}!");
    }

    new_subnet_node
        .block_on_bash_script(&format!(
            r#"
                set -e
                tar -xf {TMP_IC_REGISTRY_LOCAL_STORE_TARBALL_PATH} --zstd -C /tmp
                sudo chown -R ic-replica:ic-registry-local-store /tmp/ic_registry_local_store
                sudo chown ic-replica {TMP_NNS_PUBLIC_KEY_PATH}
                sudo systemctl stop ic-replica
                sudo rm -rf /var/lib/ic/data/ic_registry_local_store
                sudo mv /tmp/ic_registry_local_store /var/lib/ic/data/ic_registry_local_store
                sudo mv {TMP_NNS_PUBLIC_KEY_PATH} /var/lib/ic/data/nns_public_key.pem
                sudo systemctl start ic-replica
            "#
        ))
        .unwrap_or_else(|e| panic!("Could not ... on {new_subnet_node_ip:?} because {e:?}",));
    info!(
        logger,
        "Moved node {new_subnet_node_ip:?} to the recovered NNS"
    );
}

fn create_subnet(
    env: TestEnv,
    recovered_nns_node: IcNodeSnapshot,
    neuron_id: NeuronId,
    new_subnet_node: IcNodeSnapshot,
) {
    let logger = env.logger();
    let new_subnet_node_ip = new_subnet_node.get_ip_addr();
    info!(
        logger,
        "Proposing to create new subnet from {new_subnet_node_ip:?}"
    );

    let ic_admin_path = env
        .clone()
        .get_dependency_path("rs/tests/recovery/binaries/ic-admin");
    let recovered_nns_url = recovered_nns_node.get_public_url();
    let pem = env.get_dependency_path("rs/tests/nns/secret_key.pem");
    let neuron_id_number = neuron_id.0;
    let replica_version = env.get_initial_replica_version().unwrap();
    let mut cmd = Command::new(ic_admin_path);
    let new_subnet_node_id = new_subnet_node.node_id.to_string();
    cmd.arg("--nns-url")
        .arg(recovered_nns_url.to_string())
        .arg("-s")
        .arg(pem)
        .arg("propose-to-create-subnet")
        .arg("--summary")
        .arg("Creating a subnet")
        .arg("--proposer")
        .arg(neuron_id_number.to_string())
        .arg("--subnet-type")
        .arg("application")
        .arg("--replica-version-id")
        .arg(replica_version.to_string())
        .arg(new_subnet_node_id.clone());
    info!(logger, "{cmd:?} ...");
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("Could not create new subnet because {e:?}",));
    std::io::stdout().write_all(&out.stdout).unwrap();
    std::io::stderr().write_all(&out.stderr).unwrap();
    if out.status.success() {
        info!(
            logger,
            "Successfully proposed to create new subnet from {new_subnet_node_ip:?}"
        );
    } else {
        panic!("Could not create new subnet");
    }

    info!(
        logger,
        "Waiting until the new subnet with node {new_subnet_node_id} appears in the registry local store ..."
    );
    retry(
        logger.clone(),
        Duration::from_secs(500),
        Duration::from_secs(5),
        || recovered_nns_node.block_on_bash_script(
            &format!(
                r#"
                    until /opt/ic/bin/ic-regedit snapshot /var/lib/ic/data/ic_registry_local_store \
                        | jq 'to_entries | .[] | select(.key | startswith("subnet_record_")) | select(.value.membership == ["(principal-id){new_subnet_node_id}"])' \
                            --exit-status; do
                        sleep 1;
                    done
                "#
            )
        ),
    )
    .unwrap_or_else(|e| {
        panic!("Node {new_subnet_node_id} did not become a member of a new subnet in time. Error: {e:?}")
    });
}

/// Imports the test PEM as a dfx identity
/// and returns its Principal and its ledger account ID.
fn dfx_import_identity(env: TestEnv) -> PrincipalId {
    let logger = env.logger();
    info!(logger, "Creating cycles wallet ...");
    let dfx_path = env.clone().get_dependency_path("external/dfx/dfx");
    let pem = env.get_dependency_path("rs/tests/nns/secret_key.pem");
    let home = env.base_path();
    let mut cmd = Command::new(dfx_path.clone());
    cmd.env("HOME", home.clone())
        .arg("identity")
        .arg("import")
        .arg("--force")
        .arg("--storage-mode=plaintext")
        .arg("nns_test_user_dfx_identity")
        .arg(pem);
    info!(logger, "{cmd:?} ...");
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("Could not run '{cmd:?}' because {e:?}"));
    std::io::stdout().write_all(&out.stdout).unwrap();
    std::io::stderr().write_all(&out.stderr).unwrap();

    let mut cmd = Command::new(dfx_path.clone());
    cmd.env("HOME", home.clone())
        .arg("identity")
        .arg("use")
        .arg("nns_test_user_dfx_identity");
    info!(logger, "{cmd:?} ...");
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("Could not run '{cmd:?}' because {e:?}",));
    std::io::stdout().write_all(&out.stdout).unwrap();
    std::io::stderr().write_all(&out.stderr).unwrap();

    let mut cmd = Command::new(dfx_path.clone());
    cmd.env("HOME", home.clone())
        .arg("identity")
        .arg("get-principal");
    info!(logger, "{cmd:?} ...");
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("Could not run '{cmd:?}' because {e:?}",));
    std::io::stdout().write_all(&out.stdout).unwrap();
    std::io::stderr().write_all(&out.stderr).unwrap();

    let principal = String::from_utf8(out.stdout).unwrap();
    let principal = principal.trim();
    let principal: PrincipalId = PrincipalId::from_str(principal)
        .unwrap_or_else(|e| panic!("Could not parse PrincipalId because {e:?}"));
    info!(logger, "principal = {principal:?}");
    principal
}

fn get_app_subnet_id(
    env: TestEnv,
    recovered_nns_node: IcNodeSnapshot,
    new_subnet_node_id: NodeId,
) -> SubnetId {
    let logger = env.logger();

    package_registry_local_store(logger.clone(), recovered_nns_node.clone());

    let priv_key_path = env
        .get_path(SSH_AUTHORIZED_PRIV_KEYS_DIR)
        .join(SSH_USERNAME);
    let recovered_nns_node_ip = recovered_nns_node.get_ip_addr();
    let src = format!(
        "{SSH_USERNAME}@[{recovered_nns_node_ip:?}]:{TMP_IC_REGISTRY_LOCAL_STORE_TARBALL_PATH}"
    );
    let dst = env.get_path("ic_registry_local_store.tar.zst");
    let mut cmd = Command::new("scp");
    cmd.arg("-oUserKnownHostsFile=/dev/null")
        .arg("-oStrictHostKeyChecking=no")
        .arg("-i")
        .arg(priv_key_path.clone())
        .arg(src.clone())
        .arg(dst.clone());
    info!(logger, "{cmd:?} ...");
    let scp_out = cmd
        .output()
        .unwrap_or_else(|e| panic!("Could not scp {src} to {dst:?} because: {e:?}!",));
    if !scp_out.status.success() {
        std::io::stdout().write_all(&scp_out.stdout).unwrap();
        std::io::stderr().write_all(&scp_out.stderr).unwrap();
        panic!("Could not scp {src} to {dst:?}!");
    }

    let recovered_nns_prep_state_dir = env
        .create_prep_dir(RECOVERED_NNS)
        .unwrap_or_else(|e| panic!("Could not create prep_dir for {RECOVERED_NNS} because {e:?}"));
    let recovered_nns_prep_state_dir_path = recovered_nns_prep_state_dir.clone().prep_dir;
    let mut cmd = Command::new("tar");
    cmd.arg("-xf")
        .arg(dst)
        .arg("--zstd")
        .arg("-C")
        .arg(recovered_nns_prep_state_dir_path.clone());
    info!(logger, "{cmd:?} ...");
    let tar_out = cmd.output().unwrap_or_else(|e| {
        panic!("Could not unpack the recovered NNS registry local store because {e:?}",)
    });
    if !tar_out.status.success() {
        std::io::stdout().write_all(&tar_out.stdout).unwrap();
        std::io::stderr().write_all(&tar_out.stderr).unwrap();
        panic!("Could not unpack the recovered NNS registry local store");
    }
    let recovered_nns_registry_local_store_path =
        recovered_nns_prep_state_dir.registry_local_store_path();
    fs::rename(
        recovered_nns_prep_state_dir_path.join("ic_registry_local_store"),
        recovered_nns_registry_local_store_path,
    )
    .unwrap_or_else(|e| panic!("Could not rename ic_registry_local_store because {e:?}"));

    let app_subnet = env
        .topology_snapshot_by_name(RECOVERED_NNS)
        .subnets()
        .find(|subnet| {
            subnet.subnet_type() == SubnetType::Application
                && subnet
                    .nodes()
                    .any(|node| node.node_id == new_subnet_node_id)
        })
        .unwrap();
    app_subnet.subnet_id
}

fn set_default_subnets(
    env: TestEnv,
    recovered_nns_node: IcNodeSnapshot,
    neuron_id: NeuronId,
    subnet_id: SubnetId,
) {
    let logger = env.logger();
    info!(logger, "Setting authorized subnetworks {subnet_id:?} ...");
    submit_execute_await_proposal(
        env,
        recovered_nns_node,
        neuron_id,
        NnsFunction::SetAuthorizedSubnetworks,
        SetAuthorizedSubnetworkListArgs {
            who: None,
            subnets: vec![subnet_id],
        },
        "Setting authorized subnetworks".to_string(),
    );
    info!(logger, "Set authorized subnetworks {subnet_id:?}.");
}

fn create_cycles_wallet(
    env: TestEnv,
    recovered_nns_node: IcNodeSnapshot,
    principal: PrincipalId,
    new_subnet_node_url: Url,
) -> CanisterId {
    let logger = env.logger();
    let dfx_path = env.clone().get_dependency_path("external/dfx/dfx");
    let recovered_nns_url = recovered_nns_node.get_public_url();
    let home = env.base_path();
    let mut cmd = Command::new(dfx_path.clone());
    cmd.env("HOME", home.clone())
        .arg("ledger")
        .arg("--network")
        .arg(recovered_nns_url.to_string())
        .arg("create-canister")
        .arg(principal.to_string())
        .arg("--amount")
        .arg("10000");
    info!(logger, "{cmd:?} ...");
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("Could not run '{cmd:?}' because {e:?}",));
    std::io::stdout().write_all(&out.stdout).unwrap();
    std::io::stderr().write_all(&out.stderr).unwrap();

    let prefix: &str = "Canister created with id: \"";
    let wallet_canister_id: CanisterId = match std::str::from_utf8(&out.stdout)
        .unwrap()
        .split('\n')
        .filter(|line| line.starts_with(prefix))
        .collect::<Vec<&str>>()
        .first()
        .unwrap()
        .split(prefix)
        .collect::<Vec<&str>>()[..]
    {
        [_, canister_id_str] => {
            let mut canister_id_string = canister_id_str.to_string();
            canister_id_string.pop(); // Drop the last " character
            CanisterId::from_str(&canister_id_string).unwrap()
        }
        _ => panic!("Couldn't parse canister ID!"),
    };
    info!(logger, "WALLET_CANISTER = {wallet_canister_id}");

    let mut cmd = Command::new(dfx_path.clone());
    cmd.env("HOME", home.clone())
        .arg("-q")
        .arg("identity")
        .arg("--network")
        .arg(new_subnet_node_url.to_string())
        .arg("deploy-wallet")
        .arg(wallet_canister_id.to_string());
    info!(logger, "{cmd:?} ...");
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("Could not run '{cmd:?}' because {e:?}",));
    std::io::stdout().write_all(&out.stdout).unwrap();
    std::io::stderr().write_all(&out.stderr).unwrap();

    let mut cmd = Command::new(dfx_path);
    cmd.env("HOME", home)
        .arg("-q")
        .arg("identity")
        .arg("--network")
        .arg(new_subnet_node_url.to_string())
        .arg("set-wallet")
        .arg(wallet_canister_id.to_string());
    info!(logger, "{cmd:?} ...");
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("Could not run '{cmd:?}' because {e:?}",));
    std::io::stdout().write_all(&out.stdout).unwrap();
    std::io::stderr().write_all(&out.stderr).unwrap();

    wallet_canister_id
}

fn configure_sns_wasms(
    env: TestEnv,
    neuron_id: NeuronId,
    recovered_nns_node: IcNodeSnapshot,
    new_subnet_id: SubnetId,
    wallet_canister_id: CanisterId,
) {
    let logger = env.logger();
    let recovered_nns_node_url = recovered_nns_node.get_public_url();

    let current_subnets = block_on(async {
        let nns = runtime_from_url(
            recovered_nns_node_url,
            PrincipalId::from_str(MAINNET_SNS_WASM_CANISTER_ID).unwrap(),
        );
        let sns_wasm_canister = Canister::new(
            &nns,
            CanisterId::from_str(MAINNET_SNS_WASM_CANISTER_ID).unwrap(),
        );
        info!(logger, "get_sns_subnet_ids ...");
        let get_sns_subnet_ids_response: GetSnsSubnetIdsResponse = sns_wasm_canister
            .query_("get_sns_subnet_ids", candid_one, GetSnsSubnetIdsRequest {})
            .await
            .unwrap();
        get_sns_subnet_ids_response.sns_subnet_ids
    });

    info!(
        logger,
        "Proposing to remove SNS subnet IDs {current_subnets:?} and add {new_subnet_id:?} ..."
    );
    submit_execute_await_proposal(
        env.clone(),
        recovered_nns_node.clone(),
        neuron_id,
        NnsFunction::UpdateSnsWasmSnsSubnetIds,
        UpdateSnsSubnetListRequest {
            sns_subnet_ids_to_add: vec![new_subnet_id.get()],
            sns_subnet_ids_to_remove: current_subnets,
        },
        "Add SNS Subnet IDs to SNS-WASM".to_string(),
    );

    info!(
        logger,
        "Proposing to allow the wallet canister {wallet_canister_id:?} to deploy an SNS ..."
    );
    submit_execute_await_proposal(
        env,
        recovered_nns_node,
        neuron_id,
        NnsFunction::UpdateAllowedPrincipals,
        UpdateAllowedPrincipalsRequest {
            added_principals: vec![wallet_canister_id.get()],
            removed_principals: vec![],
        },
        "Update the list of Principals allowed to deploy an SNS".to_string(),
    );
}

/// Write a shell script containing some environment variable exports.
/// This script can be sourced such that we can easily use the legacy
/// nns-tools shell scripts in /testnet/tools/nns-tools/ with the dynamic
/// testnet deployed by this system-test.
fn write_sh_lib(
    env: TestEnv,
    neuron_id: NeuronId,
    new_subnet_node: IcNodeSnapshot,
    wallet_canister_id: CanisterId,
    boundary_node_url: Url,
) {
    let logger: slog::Logger = env.clone().logger();
    let set_testnet_env_vars_sh_path = env.get_path(SET_TESTNET_ENV_VARS_SH);
    let set_testnet_env_vars_sh_str = set_testnet_env_vars_sh_path.display();
    let ic_admin =
        fs::canonicalize(env.get_dependency_path("rs/tests/recovery/binaries/ic-admin")).unwrap();
    let sns_cli = fs::canonicalize(env.get_dependency_path("rs/sns/cli/sns")).unwrap();
    let pem = fs::canonicalize(env.get_dependency_path("rs/tests/nns/secret_key.pem")).unwrap();
    let new_subnet_node_url = new_subnet_node.get_public_url();
    let neuron_id_number = neuron_id.0;
    let wallet_canister_id_str = wallet_canister_id.to_string();
    let sns_quill =
        fs::canonicalize(env.get_dependency_path("external/sns_quill/sns-quill")).unwrap();
    let idl2json = fs::canonicalize(env.get_dependency_path("external/idl2json/idl2json")).unwrap();
    let dfx_home = fs::canonicalize(env.base_path()).unwrap();
    fs::write(
        set_testnet_env_vars_sh_path.clone(),
        format!(
            "export IC_ADMIN={ic_admin:?};\n\
             export SNS_CLI={sns_cli:?};\n\
             export PEM={pem:?};\n\
             export NNS_URL=\"{boundary_node_url}\";\n\
             export NEURON_ID={neuron_id_number:?};\n\
             export SUBNET_URL=\"{new_subnet_node_url}\";\n\
             export WALLET_CANISTER=\"{wallet_canister_id_str}\";\n\
             export SNS_QUILL={sns_quill:?};\n\
             export IDL2JSON={idl2json:?};\n\
             export DFX_HOME={dfx_home:?};\n\
            "
        ),
    )
    .unwrap_or_else(|e| {
        panic!(
            "Writing {set_testnet_env_vars_sh_str} failed because: {}",
            e
        )
    });
    let canonical_sh_lib_path = fs::canonicalize(set_testnet_env_vars_sh_path.clone()).unwrap();
    info!(logger, "source {canonical_sh_lib_path:?}");
}

fn bless_replica_version(
    env: TestEnv,
    neuron_id: NeuronId,
    proposal_sender: Sender,
    nns_node: IcNodeSnapshot,
    replica_version: String,
) {
    info!(
        env.logger(),
        "Begin Bless replica version {}", replica_version
    );

    let logger = env.logger();
    let nns_runtime = runtime_from_url(nns_node.get_public_url(), nns_node.effective_canister_id());
    let governance_canister = get_governance_canister(&nns_runtime);
    let sha256 = env.get_ic_os_update_img_sha256().unwrap();
    let upgrade_url = env.get_ic_os_update_img_url().unwrap();

    let proposal_id = {
        let logger = logger.clone();
        let replica_version = replica_version.clone();
        block_on(async move {
            let proposal_id = submit_update_elected_replica_versions_proposal(
                &governance_canister,
                proposal_sender,
                neuron_id,
                ReplicaVersion::try_from(replica_version.clone()).unwrap(),
                sha256,
                vec![upgrade_url.to_string()],
                vec![],
            )
            .await;
            info!(
                logger,
                "Proposal {:?} to bless replica version {:?} has been submitted",
                proposal_id.to_string(),
                replica_version,
            );
            vote_execute_proposal_assert_executed(&governance_canister, proposal_id).await;
            proposal_id
        })
    };

    info!(
        logger,
        "SUCCESS! Proposal {:?} to bless replica version {:?} has been executed successfully using neuron {:?}",
        proposal_id.to_string(),
        replica_version,
        neuron_id,
    );
}

fn submit_execute_await_proposal(
    env: TestEnv,
    nns_node: IcNodeSnapshot,
    neuron_id: NeuronId,
    nns_function: NnsFunction,
    nns_function_input: impl CandidType,
    title: String,
) {
    let logger = env.logger();
    block_on(async {
        let nns = runtime_from_url(nns_node.get_public_url(), nns_node.effective_canister_id());
        let governance_canister_id = CanisterId::from_str(MAINNET_GOVERNANCE_CANISTER_ID).unwrap();
        let governance = get_canister(&nns, governance_canister_id);

        let pem_path = env.get_dependency_path("rs/tests/nns/secret_key.pem");
        let mut pem_file = File::open(pem_path).unwrap();
        let mut pem = String::new();
        pem_file.read_to_string(&mut pem).unwrap();
        let ed_25519_key_pair = Ed25519KeyPair::from_pem(&pem).unwrap();
        let sender = Sender::from_keypair(&ed_25519_key_pair);

        info!(logger, "Submitting proposal \"{title}\" ... ");
        let proposal_id = submit_external_update_proposal(
            &governance,
            sender,
            neuron_id,
            nns_function,
            nns_function_input,
            title.clone(),
            title.clone(),
        )
        .await;

        info!(
            logger,
            "Submitted proposal \"{title}\" with ID {proposal_id:?}. Executing it ..."
        );
        vote_execute_proposal_assert_executed(&governance, proposal_id).await;

        let retry_delay = Duration::from_secs(5);
        let timeout = Duration::from_secs(60);
        await_proposal_execution(&logger, &governance, proposal_id, retry_delay, timeout).await;
        info!(
            logger,
            "Executed proposal \"{title}\" with ID {proposal_id:?}"
        );
    });
}
