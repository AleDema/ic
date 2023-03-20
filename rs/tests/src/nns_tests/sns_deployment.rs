use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;
use std::time::Instant;

use crate::canister_agent::{CanisterAgent, HasCanisterAgentCapability};
use crate::canister_api::{
    CallMode, CanisterHttpRequestProvider, Icrc1RequestProvider, Icrc1TransferRequest,
    NnsDappRequestProvider, Request, Response, SnsRequestProvider,
};
use crate::canister_requests;
use crate::driver::farm::HostFeature;
use crate::driver::prometheus_vm::{HasPrometheus, PrometheusVm};
use crate::driver::test_env::TestEnv;
use crate::driver::test_env_api::TEST_USER1_STARTING_TOKENS;
use crate::driver::test_env_api::{
    GetFirstHealthyNodeSnapshot, HasGroupSetup, HasPublicApiUrl, HasTopologySnapshot,
    NnsCustomizations,
};
use crate::generic_workload_engine::engine::Engine;
use crate::generic_workload_engine::metrics::{LoadTestMetrics, LoadTestOutcome, RequestOutcome};
use candid::{Nat, Principal};
use ic_agent::{Identity, Signature};
use ic_base_types::PrincipalId;
use ic_canister_client_sender::ed25519_public_key_to_der;
use ic_ledger_core::Tokens;
use ic_nervous_system_common::E8;
use ic_rosetta_api::models::RosettaSupportedKeyPair;
use ic_rosetta_test_utils::EdKeypair;
use ic_sns_swap::pb::v1::Lifecycle;
use ic_sns_swap::swap::principal_to_subaccount;
use ic_types::Height;
use icp_ledger::{AccountIdentifier, Subaccount};
use icrc_ledger_agent::Icrc1Agent;
use icrc_ledger_types::transaction::TransferArg;
use icrc_ledger_types::Account;
use serde::{Deserialize, Serialize};
use slog::info;
use tokio::runtime::Builder;

use crate::orchestrator::utils::rw_message::{
    install_nns_and_check_progress, install_nns_with_customizations_and_check_progress,
};
use crate::sns_client::{SnsClient, SNS_SALE_PARAM_MIN_PARTICIPANT_ICP_E8S};
use crate::util::{assert_create_agent_with_identity, block_on};

use crate::driver::ic::{
    AmountOfMemoryKiB, ImageSizeGiB, InternetComputer, NrOfVCPUs, Subnet, VmResources,
};
use crate::driver::test_env::TestEnvAttribute;

use ic_nervous_system_common_test_keys::{TEST_USER1_KEYPAIR, TEST_USER1_PRINCIPAL};
use ic_nns_constants::LEDGER_CANISTER_ID;
use ic_registry_subnet_type::SubnetType;

use super::sns_aggregator::AggregatorClient;

const WORKLOAD_GENERATION_DURATION: Duration = Duration::from_secs(60);

const DKG_INTERVAL: u64 = 199;
const SUBNET_SIZE: usize = 4;
const UVM_NUM_CPUS: NrOfVCPUs = NrOfVCPUs::new(2);
const UVM_MEMORY_SIZE: AmountOfMemoryKiB = AmountOfMemoryKiB::new(67108864); // 64 GiB
const UVM_BOOT_IMAGE_MIN_SIZE: ImageSizeGiB = ImageSizeGiB::new(4);

const REQUESTS_DISPATCH_EXTRA_TIMEOUT: Duration = Duration::from_secs(1_000);

// This constant is simply an encoding of a CanisterId(x) for some small value of x.
// x is the position of the sale (a.k.a. swap) canister in the SNS application subnet.
const SNS_SALE_CANISTER_ID: &str = "5j7vn-7yaaa-aaaaa-qaaca-cai";

pub const NUM_SNS_SALE_PARTICIPANTS: usize = 100;

pub fn setup_static_testnet(env: TestEnv) {
    SnsClient::get_sns_client_for_static_testnet(&env);
}

pub fn workload_static_testnet_fe_users(env: TestEnv) {
    let log = env.logger();
    let duration =
        std::env::var("DURATION_MINUTES").expect("variable DURATION_MINUTES not specified");
    let duration: usize = duration
        .parse()
        .unwrap_or_else(|_| panic!("cannot parse as usize: `{duration}`"));
    let duration = Duration::from_secs(duration as u64 * 60);

    let aggr_canister_id =
        std::env::var("AGGREGATOR_CANISTER").expect("variable AGGREGATOR_CANISTER not specified");
    let aggr_canister_id = Principal::from_text(aggr_canister_id).unwrap();
    let aggregator_sns_request_provider = CanisterHttpRequestProvider::new(aggr_canister_id);
    let static_testnet_name = std::env::var("TESTNET").expect("variable TESTNET not specified");
    let static_testnet_bn_url = &format!("https://{static_testnet_name}.testnet.dfinity.network/");

    let icp_ledger_provider = Icrc1RequestProvider::new_icp_ledger_request_provider();
    let sns_request_provider = SnsRequestProvider::from_env(&env);

    let account = {
        let sns_client = SnsClient::read_attribute(&env);
        let sns_sale_canister_id = sns_client.sns_canisters.swap().get();
        let sns_subaccount = Some(principal_to_subaccount(&TEST_USER1_PRINCIPAL));
        Account {
            owner: sns_sale_canister_id.0,
            subaccount: sns_subaccount,
        }
    };

    let nns_dapp_canister_id =
        std::env::var("NNS_DAPP_CANISTER").expect("variable NNS_DAPP_CANISTER not specified");
    let nns_dapp_canister_id = Principal::from_text(nns_dapp_canister_id).unwrap();
    let nns_dapp_request_provider = NnsDappRequestProvider::new(nns_dapp_canister_id);
    let buyer = Some(*TEST_USER1_PRINCIPAL);

    let large_asset_name =
        std::env::var("LARGE_ASSET_NAME").expect("variable LARGE_ASSET_NAME not specified");

    let num_requests: usize = 48;

    let future_generator = {
        let agent = block_on(CanisterAgent::from_boundary_node_url(static_testnet_bn_url));

        move |idx| {
            let agent = agent.clone();
            let large_asset_name = large_asset_name.clone();
            async move {
                let agent = agent.clone();
                let request_outcome = canister_requests![
                    idx,
                    1 * agent => icp_ledger_provider.icrc1_balance_of_request(account, CallMode::Query),
                    1 * agent => icp_ledger_provider.icrc1_balance_of_request(account, CallMode::Update),
                    1 * agent => nns_dapp_request_provider.get_account_request("abc".to_string(), CallMode::Query),
                    1 * agent => nns_dapp_request_provider.get_account_request("abc".to_string(), CallMode::Update),
                    1 * agent => sns_request_provider.get_buyer_state(buyer, CallMode::Query),
                    1 * agent => sns_request_provider.get_buyer_state(buyer, CallMode::Update),
                    1 * agent => aggregator_sns_request_provider.http_request(AggregatorClient::aggregator_http_endpoint()),
                    1 * agent => nns_dapp_request_provider.http_request(large_asset_name),
                    40 * agent => nns_dapp_request_provider.http_request("/main.js".to_string()),
                ];
                request_outcome.into_test_outcome()
            }
        }
    };

    // Compute the raw RPS based on the effective RPS specified by the user
    let effective_rps = std::env::var("WORKLOAD_RPS").expect("variable WORKLOAD_RPS not specified");
    let effective_rps: usize = effective_rps
        .parse()
        .unwrap_or_else(|_| panic!("cannot parse as usize: `{effective_rps}`"));
    let raw_rps = effective_rps * num_requests;

    // --- Generate workload ---
    let workload = Engine::new(log.clone(), future_generator, raw_rps, duration)
        .increase_dispatch_timeout(REQUESTS_DISPATCH_EXTRA_TIMEOUT);

    let metrics = {
        let aggr = LoadTestMetrics::new(log);
        let fun = LoadTestMetrics::aggregator_fn;
        block_on(workload.execute(aggr, fun)).expect("Workload execution has failed.")
    };

    env.emit_report(format!("{metrics}"));
}

pub fn workload_static_testnet_get_account(env: TestEnv) {
    let log = env.logger();
    let duration =
        std::env::var("DURATION_MINUTES").expect("variable DURATION_MINUTES not specified");
    let duration: usize = duration
        .parse()
        .unwrap_or_else(|_| panic!("cannot parse as usize: `{duration}`"));
    let duration = Duration::from_secs(duration as u64 * 60);

    let static_testnet_name = std::env::var("TESTNET").expect("variable TESTNET not specified");
    let static_testnet_bn_url = &format!("https://{static_testnet_name}.testnet.dfinity.network/");

    let nns_dapp_canister_id =
        std::env::var("NNS_DAPP_CANISTER").expect("variable NNS_DAPP_CANISTER not specified");
    let nns_dapp_canister_id = Principal::from_text(nns_dapp_canister_id).unwrap();
    let nns_dapp_request_provider = NnsDappRequestProvider::new(nns_dapp_canister_id);

    // Compute the raw RPS based on the effective RPS specified by the user
    let rps = std::env::var("WORKLOAD_RPS").expect("variable WORKLOAD_RPS not specified");
    let rps: usize = rps
        .parse()
        .unwrap_or_else(|_| panic!("cannot parse as usize: `{rps}`"));

    let future_generator = {
        let agent = block_on(CanisterAgent::from_boundary_node_url(static_testnet_bn_url));
        move |_idx| {
            let agent = agent.clone();
            async move {
                let agent = agent.clone();
                let request = nns_dapp_request_provider
                    .get_account_request("test_account".to_string(), CallMode::Update);
                agent.call(&request).await.map(|_| ()).into_test_outcome()
            }
        }
    };

    // --- Generate workload ---
    let workload = Engine::new(log.clone(), future_generator, rps, duration)
        .increase_dispatch_timeout(REQUESTS_DISPATCH_EXTRA_TIMEOUT);

    let metrics = {
        let aggr = LoadTestMetrics::new(log);
        let fun = LoadTestMetrics::aggregator_fn;
        block_on(workload.execute(aggr, fun)).expect("Workload execution has failed.")
    };
    env.emit_report(format!("{metrics}"));
}

pub fn workload_static_testnet_sale_bot(env: TestEnv) {
    let log = env.logger();
    let duration =
        std::env::var("DURATION_MINUTES").expect("variable DURATION_MINUTES not specified");
    let duration: usize = duration
        .parse()
        .unwrap_or_else(|_| panic!("cannot parse as usize: `{duration}`"));
    let duration = Duration::from_secs(duration as u64 * 60);

    let static_testnet_name = std::env::var("TESTNET").expect("variable TESTNET not specified");
    let static_testnet_bn_url = &format!("https://{static_testnet_name}.testnet.dfinity.network/");

    let sns_request_provider = SnsRequestProvider::from_env(&env);

    // Compute the raw RPS based on the effective RPS specified by the user
    let rps = std::env::var("WORKLOAD_RPS").expect("variable WORKLOAD_RPS not specified");
    let rps: usize = rps
        .parse()
        .unwrap_or_else(|_| panic!("cannot parse as usize: `{rps}`"));

    let future_generator = {
        let agent = block_on(CanisterAgent::from_boundary_node_url(static_testnet_bn_url));
        move |_idx| {
            let agent = agent.clone();
            async move {
                let agent = agent.clone();
                let request = sns_request_provider.refresh_buyer_tokens(None);
                agent.call(&request).await.map(|_| ()).into_test_outcome()
            }
        }
    };

    // --- Generate workload ---
    let workload = Engine::new(log.clone(), future_generator, rps, duration)
        .increase_dispatch_timeout(REQUESTS_DISPATCH_EXTRA_TIMEOUT);

    let metrics = {
        let aggr = LoadTestMetrics::new(log);
        let fun = LoadTestMetrics::aggregator_fn;
        block_on(workload.execute(aggr, fun)).expect("Workload execution has failed.")
    };
    env.emit_report(format!("{metrics}"));
}

fn setup(env: TestEnv, sale_participants: Vec<SaleParticipant>) {
    env.ensure_group_setup_created();

    PrometheusVm::default()
        .start(&env)
        .expect("failed to start prometheus VM");

    InternetComputer::new()
        .with_required_host_features(vec![HostFeature::SnsLoadTest])
        .with_default_vm_resources(VmResources {
            vcpus: Some(UVM_NUM_CPUS),
            memory_kibibytes: Some(UVM_MEMORY_SIZE),
            boot_image_minimal_size_gibibytes: Some(UVM_BOOT_IMAGE_MIN_SIZE),
        })
        .add_subnet(
            Subnet::new(SubnetType::System)
                .with_dkg_interval_length(Height::from(DKG_INTERVAL))
                .add_nodes(SUBNET_SIZE),
        )
        .add_subnet(
            Subnet::new(SubnetType::Application)
                .with_dkg_interval_length(Height::from(DKG_INTERVAL))
                .add_nodes(SUBNET_SIZE),
        )
        .setup_and_start(&env)
        .expect("failed to setup IC under test");

    env.sync_prometheus_config_with_topology();

    // Set up the initial ICP ledger
    let mut ledger_balances = HashMap::new();
    for participant in sale_participants {
        if participant.starting_sns_balance.get_e8s() > 0 {
            let account_identifier = participant.sns_account_identifier();
            ledger_balances.insert(account_identifier, participant.starting_sns_balance);
        }
        if participant.starting_icp_balance.get_e8s() > 0 {
            let account_identifier = participant.icp_account_identifier();
            ledger_balances.insert(account_identifier, participant.starting_icp_balance);
        }
    }
    let nns_customizations = NnsCustomizations {
        ledger_balances: Some(ledger_balances),
    };

    // Install NNS with ledger customizations
    install_nns(&env, Some(nns_customizations));

    install_sns(&env);
}

pub fn sns_setup(env: TestEnv) {
    setup(env, vec![]);
}

pub fn sns_setup_with_many_sale_participants(env: TestEnv) {
    // Generate random identities for all the participants
    let participants: Vec<SaleParticipant> = (1..NUM_SNS_SALE_PARTICIPANTS + 1)
        .map(|x| {
            let name = format!("user_{x}");
            let starting_icp_balance = Tokens::ZERO;
            // The amount of ICPs in this user's SNS sale sub-account is minimally enough for sale participation.
            let starting_sns_balance = Tokens::from_e8s(SNS_SALE_PARAM_MIN_PARTICIPANT_ICP_E8S);
            let seed = x as u64;
            SaleParticipant::random(name, starting_icp_balance, starting_sns_balance, seed)
        })
        .collect();

    // Make sure these participants are available after the setup
    participants.write_attribute(&env);

    // Run the actual setup
    setup(env, participants);
}

pub fn sns_setup_with_many_icp_users(env: TestEnv) {
    // Generate random identities for all the participants
    let participants: Vec<SaleParticipant> = (1..NUM_SNS_SALE_PARTICIPANTS + 1)
        .map(|x| {
            let name = format!("user_{x}");
            // The amount of ICPs in this user's default ICP account should suffise for at least 60 minimal SNS sale transfers + transfer fees.
            let starting_icp_balance =
                Tokens::from_e8s(1_200 * SNS_SALE_PARAM_MIN_PARTICIPANT_ICP_E8S);
            let starting_sns_balance = Tokens::ZERO;
            let seed = x as u64;
            SaleParticipant::random(name, starting_icp_balance, starting_sns_balance, seed)
        })
        .collect();

    // Make sure these participants are available after the setup
    participants.write_attribute(&env);

    // Run the actual setup
    setup(env, participants);
}

pub fn init_participants(env: TestEnv) {
    let log = env.logger();
    let start_time = Instant::now();
    let participants = Vec::<SaleParticipant>::read_attribute(&env);
    let participants_str: Vec<String> = participants.iter().map(|p| p.name.clone()).collect();
    let sns_client = SnsClient::read_attribute(&env);
    let app_node = env.get_first_healthy_application_node_snapshot();
    let sns_request_provider = SnsRequestProvider::from_sns_client(&sns_client);
    block_on(async move {
        for participant in participants {
            let canister_agent = app_node
                .build_canister_agent_with_identity(participant.clone())
                .await;
            let request = sns_request_provider.refresh_buyer_tokens(Some(participant.principal_id));
            info!(
                log,
                "Submitting request {request:?} from {:?} ...", participant.principal_id
            );
            let res = canister_agent
                .call_and_parse(&request)
                .await
                .result()
                .unwrap();
            info!(
                log,
                "Update call from {} to `sns_sale.refresh_buyer_tokens` returned {res:?} (elapsed {:?})",
                participant.name,
                start_time.elapsed()
            );
        }
    });
    info!(
        env.logger(),
        "==== Successfully added {} participants ({:?}) to the token sale (elapsed {:?}) ====",
        participants_str.len(),
        participants_str,
        start_time.elapsed()
    );
}

pub fn check_all_participants(env: TestEnv) {
    let log = env.logger();
    let start_time = Instant::now();
    let participants = Vec::<SaleParticipant>::read_attribute(&env);
    let participants_str: Vec<String> = participants.iter().map(|p| p.name.clone()).collect();
    let sns_client = SnsClient::read_attribute(&env);
    let sns_request_provider = SnsRequestProvider::from_sns_client(&sns_client);
    let app_node = env.get_first_healthy_application_node_snapshot();
    block_on(async move {
        for participant in participants {
            let canister_agent = app_node
                .build_canister_agent_with_identity(participant.clone())
                .await;
            let request = sns_request_provider
                .get_buyer_state(Some(participant.principal_id), CallMode::Query);
            info!(log, "Submitting request {request:?} ...");
            let res = canister_agent
                .call_and_parse(&request)
                .await
                .result()
                .unwrap();
            info!(
                log,
                "Query call from {} to `sns_sale.get_buyer_state` returned {res:?} (elapsed {:?})",
                participant.name,
                start_time.elapsed()
            );
            assert!(res.buyer_state.is_some());
        }
    });
    info!(
        env.logger(),
        "==== Successfully checked {} participants ({:?}) to the token sale (elapsed {:?}) ====",
        participants_str.len(),
        participants_str,
        start_time.elapsed()
    );
}

pub fn install_nns(env: &TestEnv, customizations: Option<NnsCustomizations>) {
    let log = env.logger();
    let start_time = Instant::now();
    if let Some(customizations) = customizations {
        install_nns_with_customizations_and_check_progress(env.topology_snapshot(), customizations);
    } else {
        install_nns_and_check_progress(env.topology_snapshot());
    }
    info!(
        log,
        "=========== The NNS has been successfully installed in {:?} ==========",
        start_time.elapsed()
    );
}

pub fn install_sns(env: &TestEnv) {
    let log = env.logger();
    let start_time = Instant::now();
    let sns_client = SnsClient::install_sns_and_check_healthy(env);
    {
        let observed = sns_client.sns_canisters.swap().get();
        let expected = PrincipalId::from_str(SNS_SALE_CANISTER_ID)
            .expect("cannot parse PrincipalId of the SNS sale (a.k.a. swap) canister");
        assert_eq!(
            observed, expected,
            "SNS sale canister got unexpected PrincipalId {observed:?} (expected {expected:?}"
        );
    }
    info!(
        log,
        "========== The SNS has been installed successfully in {:?} ===========",
        start_time.elapsed()
    );
}

pub fn initiate_token_swap(env: TestEnv) {
    let log = env.logger();
    let start_time = Instant::now();

    let sns_client = SnsClient::read_attribute(&env);
    sns_client.initiate_token_swap(&env);
    sns_client.assert_state(&env, Lifecycle::Open);
    info!(
        log,
        "==== The SNS token sale has been initialized successfully in {:?} ====",
        start_time.elapsed()
    );
}

pub fn workload_many_users_rps20_refresh_buyer_tokens(env: TestEnv) {
    let request = SnsRequestProvider::from_env(&env).refresh_buyer_tokens(None);
    let rps: usize = 20;
    generate_sns_workload_with_many_users(env, rps, Duration::from_secs(10), request);
}

pub fn workload_many_users_rps100_refresh_buyer_tokens(env: TestEnv) {
    let request = SnsRequestProvider::from_env(&env).refresh_buyer_tokens(None);
    let rps: usize = 100;
    generate_sns_workload_with_many_users(env, rps, WORKLOAD_GENERATION_DURATION, request);
}

pub fn workload_many_users_rps200_refresh_buyer_tokens(env: TestEnv) {
    let request = SnsRequestProvider::from_env(&env).refresh_buyer_tokens(None);
    let rps: usize = 200;
    generate_sns_workload_with_many_users(env, rps, WORKLOAD_GENERATION_DURATION, request);
}

pub fn workload_many_users_rps400_refresh_buyer_tokens(env: TestEnv) {
    let request = SnsRequestProvider::from_env(&env).refresh_buyer_tokens(None);
    let rps: usize = 400;
    generate_sns_workload_with_many_users(env, rps, WORKLOAD_GENERATION_DURATION, request);
}

pub fn generate_sns_workload_with_many_users<T, R>(
    env: TestEnv,
    rps: usize,
    duration: Duration,
    request: R,
) where
    T: Response,
    R: Request<T> + Clone + Sync + Send + 'static,
{
    let log = env.logger();

    let future_generator = {
        let app_node = env.get_first_healthy_application_node_snapshot();
        let agents: Vec<(SaleParticipant, CanisterAgent)> =
            Vec::<SaleParticipant>::read_attribute(&env)
                .into_iter()
                .map(|p| {
                    let canister_agent =
                        block_on(app_node.build_canister_agent_with_identity(p.clone()));
                    (p, canister_agent)
                })
                .collect();
        move |idx| {
            let request = request.clone();
            let (_, agent): &(_, CanisterAgent) = &agents[idx % agents.len()];
            let agent = agent.clone();
            async move {
                let request = request.clone();
                let agent = agent.clone();
                let fut = agent.call(&request);
                fut.await.map(|_| ()).into_test_outcome()
            }
        }
    };

    let workload = Engine::new(log.clone(), future_generator, rps, duration)
        .increase_dispatch_timeout(REQUESTS_DISPATCH_EXTRA_TIMEOUT);
    let metrics = {
        let aggr = LoadTestMetrics::new(log);
        let fun = LoadTestMetrics::aggregator_fn;
        block_on(workload.execute(aggr, fun)).expect("Workload execution has failed.")
    };
    env.emit_report(format!("{metrics}"));
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnsUsers {
    participants: Vec<SaleParticipant>,
}

impl TestEnvAttribute for SnsUsers {
    fn attribute_name() -> String {
        "sns_users".to_string()
    }
}

/// An SNS sale participant.
/// Warning: This type should be used for testing purposes only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaleParticipant {
    name: String,
    secret_key: [u8; 32],
    public_key: [u8; 32],
    principal_id: PrincipalId,
    starting_icp_balance: Tokens,
    starting_sns_balance: Tokens,
}

impl TestEnvAttribute for Vec<SaleParticipant> {
    fn attribute_name() -> String {
        "sns_sale_participants".to_string()
    }
}

impl SaleParticipant {
    pub fn random(
        name: String,
        starting_icp_balance: Tokens,
        starting_sns_balance: Tokens,
        seed: u64,
    ) -> Self {
        let key_pair = EdKeypair::generate_from_u64(seed);
        let principal_id = key_pair.generate_principal_id().unwrap();
        Self {
            name,
            principal_id,
            secret_key: key_pair.secret_key,
            public_key: key_pair.public_key,
            starting_sns_balance,
            starting_icp_balance,
        }
    }

    pub fn key_pair(&self) -> EdKeypair {
        EdKeypair {
            secret_key: self.secret_key,
            public_key: self.public_key,
        }
    }

    pub fn icp_account(&self) -> Account {
        Account {
            owner: self.principal_id.into(),
            subaccount: None,
        }
    }

    pub fn icp_account_identifier(&self) -> AccountIdentifier {
        AccountIdentifier::from(self.icp_account())
    }

    pub fn sns_account(&self) -> Account {
        let owner = PrincipalId::from_str(SNS_SALE_CANISTER_ID)
            .expect("cannot parse PrincipalId of the SNS sale (a.k.a. swap) canister")
            .into();
        let subaccount = Some(Subaccount(principal_to_subaccount(&self.principal_id)).0);
        Account { owner, subaccount }
    }

    pub fn sns_account_identifier(&self) -> AccountIdentifier {
        AccountIdentifier::from(self.sns_account())
    }
}

impl Identity for SaleParticipant {
    fn sender(&self) -> Result<Principal, String> {
        let principal = Principal::try_from(self.principal_id).unwrap();
        Ok(principal)
    }

    fn sign(&self, msg: &[u8]) -> Result<Signature, String> {
        let signature = self.key_pair().sign(msg.as_ref());
        let pk = self.key_pair().get_pb_key();
        let pk_der = ed25519_public_key_to_der(pk);
        Ok(Signature {
            signature: Some(signature.as_ref().to_vec()),
            public_key: Some(pk_der),
        })
    }
}

/// This function tests the SNS payment flow scenario for a single user, without the ticketing system.
/// For testing the payment flow for multiple users with the ticketing system, see [`generate_ticket_participants_workload`].
pub fn add_one_participant(env: TestEnv) {
    // Runbook:
    // Our goal is to establish that the wealthy user does not initially participate in the token sale.
    // For this purpose, we submit three calls:
    //   1. refresh_buyer_tokens (update) from the default user - should return an error
    //   2. refresh_buyer_tokens (update) from the wealthy user - should return an error
    //   3. get_buyer_state (query) from the default user (should return "none" for the buyer state)
    // Afterwards, we will transfer some ICPs from this user's main account to their SNS sale subaccount.
    // Finally, we will check that the user's participate has been set up correctly after the transaction.
    // For this purpose, we submit three more calls:
    //   4. refresh_buyer_tokens (update) from the default user - should return res4
    //   5. refresh_buyer_tokens (update) from the wealthy user - should return res5; it should be that res5 == res4
    //   6. get_buyer_state (query) from the default user (should return "some" for the buyer state)

    let log = env.logger();
    let start_time = Instant::now();

    // Obtain the SNS client and the SNS request provider
    let sns_client = SnsClient::read_attribute(&env);
    let sns_request_provider = SnsRequestProvider::from_sns_client(&sns_client);

    // Set up the wealthy users' account (this one has 200_000 ICP at start; see `install_nns_canisters`)
    let wealthy_user_identity = SaleParticipant {
        name: "wealthy_sale_participant".to_string(),
        principal_id: *TEST_USER1_PRINCIPAL,
        secret_key: TEST_USER1_KEYPAIR.secret_key,
        public_key: TEST_USER1_KEYPAIR.public_key,
        starting_sns_balance: Tokens::from_tokens(0).unwrap(),
        starting_icp_balance: TEST_USER1_STARTING_TOKENS,
    };

    info!(log, "Obtaining an agent to talk to the ICP Ledger ...");
    let wealthy_ledger_agent = {
        let nns_node = env.get_first_healthy_nns_node_snapshot();
        let agent = block_on(assert_create_agent_with_identity(
            nns_node.get_public_url().as_str(),
            wealthy_user_identity.clone(),
        ));
        let ledger_canister_id = Principal::try_from(LEDGER_CANISTER_ID.get()).unwrap();
        Icrc1Agent {
            agent,
            ledger_canister_id,
        }
    };
    info!(
        log,
        "Obtaining two alternative agents to talk to the SNS sale canister ..."
    );
    let app_node = env.get_first_healthy_application_node_snapshot();
    let default_sns_agent = block_on(app_node.build_canister_agent());
    let wealthy_sns_agent =
        block_on(app_node.build_canister_agent_with_identity(wealthy_user_identity.clone()));
    info!(
        log,
        "All three agents are ready (elapsed {:?})",
        start_time.elapsed()
    );

    info!(log, "Checking that buyer identity is correctly set up by calling `sns_sale.refresh_buyer_tokens` in two different ways ...");
    // Use the default identity to call refresh_buyer_tokens for the wealthy user
    let res_1 = {
        let request =
            sns_request_provider.refresh_buyer_tokens(Some(wealthy_user_identity.principal_id));
        block_on(default_sns_agent.call(&request)).result()
    };
    info!(
        log,
        "First update call to `sns_sale.refresh_buyer_tokens` returned {res_1:?} (elapsed {:?})",
        start_time.elapsed()
    );
    assert!(res_1.is_err());
    // Use the wealthy user's identity refresh_buyer_tokens for "self"
    let res_2 = {
        let request = sns_request_provider.refresh_buyer_tokens(None);
        block_on(wealthy_sns_agent.call(&request)).result()
    };
    info!(
        log,
        "Second update call to `sns_sale.refresh_buyer_tokens` returned {res_2:?} (elapsed {:?})",
        start_time.elapsed()
    );
    assert!(res_2.is_err());

    info!(
        log,
        "Validating the pre-transfer state via the `get_buyer_state` endpoint ..."
    );
    let res_3 = {
        let request = sns_request_provider
            .get_buyer_state(Some(wealthy_user_identity.principal_id), CallMode::Query);
        block_on(default_sns_agent.call_and_parse(&request))
            .result()
            .unwrap()
    };
    info!(
        log,
        "Query call to `sns_sale.get_buyer_state` returned {res_3:?} (elapsed {:?})",
        start_time.elapsed()
    );
    assert!(res_3.buyer_state.is_none());
    info!(
        log,
        "Validated pre-transfer state {:?} (elapsed {:?})",
        res_3.buyer_state,
        start_time.elapsed()
    );

    info!(
        log,
        "Transferring tokens in two transactions, 3 and 2 ICP, resp"
    );
    let sns_sale_canister_id = sns_client.sns_canisters.swap().get();
    let sns_subaccount = Some(principal_to_subaccount(&wealthy_user_identity.principal_id));
    let sns_account = Account {
        owner: sns_sale_canister_id.0,
        subaccount: sns_subaccount,
    };
    let block_idx_1 = {
        let transfer_arg = TransferArg {
            from_subaccount: None,
            to: sns_account,
            fee: None,
            created_at_time: None,
            memo: None,
            amount: Nat::from(3 * E8),
        };
        let block_idx = block_on(wealthy_ledger_agent.transfer(transfer_arg))
            .unwrap()
            .unwrap();
        info!(log, "Transaction 1: from {:?} to {sns_sale_canister_id:?} (subaccount {sns_subaccount:?}) returned block_idx={block_idx:?}", wealthy_user_identity.principal_id);
        block_idx
    };
    info!(
        log,
        "First update call to `icp_ledger.transfer` returned {block_idx_1:?} (elapsed {:?})",
        start_time.elapsed()
    );
    let block_idx_2 = {
        let transfer_arg = TransferArg {
            from_subaccount: None,
            to: sns_account,
            fee: None,
            created_at_time: None,
            memo: None,
            amount: Nat::from(2 * E8),
        };
        let block_idx = block_on(wealthy_ledger_agent.transfer(transfer_arg))
            .unwrap()
            .unwrap();
        info!(log, "Transaction 2: from {:?} to {sns_sale_canister_id:?} (subaccount {sns_subaccount:?}) returned block_idx={block_idx:?}", wealthy_user_identity.principal_id);
        block_idx
    };
    info!(
        log,
        "Second update call to `icp_ledger.transfer` returned {block_idx_2:?} (elapsed {:?})",
        start_time.elapsed()
    );
    assert_eq!(block_idx_1 + 1, block_idx_2);

    info!(log, "Validating the participation setup by calling `sns_sale.refresh_buyer_tokens` in two different ways ...");
    // Use the default identity to call refresh_buyer_tokens for the wealthy user
    let res_4 = {
        let request =
            sns_request_provider.refresh_buyer_tokens(Some(wealthy_user_identity.principal_id));
        block_on(default_sns_agent.call_and_parse(&request))
            .result()
            .unwrap()
    };
    info!(
        log,
        "Third update call to `sns_sale.refresh_buyer_tokens` returned {res_4:?} (elapsed {:?})",
        start_time.elapsed()
    );
    // Use the wealthy user's identity to call refresh_buyer_tokens for "self"
    let res_5 = {
        let request = sns_request_provider.refresh_buyer_tokens(None);
        block_on(wealthy_sns_agent.call_and_parse(&request))
            .result()
            .unwrap()
    };
    info!(
        log,
        "Fourth update call to `sns_sale.refresh_buyer_tokens` returned {res_5:?} (elapsed {:?})",
        start_time.elapsed()
    );
    assert_eq!(res_4, res_5, "sns_sale.refresh_buyer_tokens(Some({:?})) = {res_4:?}, but sns_sale.refresh_buyer_tokens(None) = {res_5:?}", wealthy_user_identity.principal_id);
    info!(log, "After setting up sale participation, the response from `sns_sale.refresh_buyer_tokens` is {res_4:?}");

    info!(
        log,
        "Validating the participation setup via the `get_buyer_state` endpoint ..."
    );
    let res_6 = {
        let request = sns_request_provider
            .get_buyer_state(Some(wealthy_user_identity.principal_id), CallMode::Query);
        block_on(default_sns_agent.call_and_parse(&request))
            .result()
            .unwrap()
    };
    info!(
        log,
        "Query call to `sns_sale.get_buyer_state` returned {res_6:?} (elapsed {:?})",
        start_time.elapsed()
    );
    assert!(res_6.buyer_state.is_some());
    info!(log, "Established buyer state {:?}", res_6.buyer_state);

    info!(
        log,
        "==== Successfully added {:?} to the token sale participants (elapsed {:?}) ====",
        wealthy_user_identity.principal_id,
        start_time.elapsed()
    );
}

pub fn generate_ticket_participants_workload(env: TestEnv, rps: usize, duration: Duration) {
    let log = env.logger();
    // The amount of ICP that each participant will contribute to the SNS
    let icp_amount = SNS_SALE_PARAM_MIN_PARTICIPANT_ICP_E8S;

    let future_generator = {
        let nns_node = env.get_first_healthy_nns_node_snapshot();
        let app_node = env.get_first_healthy_application_node_snapshot();
        let sns_client = SnsClient::read_attribute(&env);
        let sns_request_provider = SnsRequestProvider::from_sns_client(&sns_client);
        let sns_sale_canister_id = sns_client.sns_canisters.swap().get();
        let ledger_canister_id = Principal::try_from(LEDGER_CANISTER_ID.get()).unwrap();
        let wealthy_ledger_agent: CanisterAgent = {
            let wealthy_user_identity = SaleParticipant {
                name: "wealthy_sale_participant".to_string(),
                principal_id: *TEST_USER1_PRINCIPAL,
                secret_key: TEST_USER1_KEYPAIR.secret_key,
                public_key: TEST_USER1_KEYPAIR.public_key,
                starting_sns_balance: Tokens::ZERO,
                starting_icp_balance: TEST_USER1_STARTING_TOKENS,
            };
            block_on(nns_node.build_canister_agent_with_identity(wealthy_user_identity))
        };
        move |idx| {
            let (nns_node, app_node, wealthy_ledger_agent) = (
                nns_node.clone(),
                app_node.clone(),
                wealthy_ledger_agent.clone(),
            );
            async move {
                let (nns_node, app_node, wealthy_ledger_agent) = (
                    nns_node.clone(),
                    app_node.clone(),
                    wealthy_ledger_agent.clone(),
                );
                let (participant, ledger_agent, canister_agent) = {
                    let name = format!("user_{idx}");
                    let starting_icp_balance = Tokens::ZERO;
                    // The amount of ICPs in this user's SNS sale sub-account is minimally enough for sale participation.
                    let starting_sns_balance =
                        Tokens::from_e8s(SNS_SALE_PARAM_MIN_PARTICIPANT_ICP_E8S);
                    // The seed should depend on all inputs of `generate_ticket_participants_workload` and this closure to avoid
                    // re-creating the same participants in subsequent calls to `generate_ticket_participants_workload`, all of which
                    // are assumed to have different values for `duration` and `rps`).
                    let seed = ((idx as u64) << 32) + (duration.as_secs() << 16) + (rps as u64);
                    let p = SaleParticipant::random(
                        name,
                        starting_icp_balance,
                        starting_sns_balance,
                        seed,
                    );
                    let ledger_agent = nns_node.build_canister_agent_with_identity(p.clone()).await;
                    let canister_agent =
                        app_node.build_canister_agent_with_identity(p.clone()).await;
                    (p, ledger_agent, canister_agent)
                };
                let sns_subaccount = Subaccount(principal_to_subaccount(&participant.principal_id));
                let mut sale_outcome = LoadTestOutcome::<(), String>::default();
                let overall_start_time = Instant::now();
                let mut overall_result: Result<(), String> = Ok(());

                // 0. "Mint" tokens
                if overall_result.is_ok() {
                    overall_result = {
                        let transfer_arg = TransferArg {
                            from_subaccount: None,
                            to: participant.icp_account(),
                            fee: None,
                            created_at_time: None,
                            memo: None,
                            amount: Nat::from(icp_amount * 2), // should cover one minimal participation + fee
                        };
                        wealthy_ledger_agent.call_and_parse(&Icrc1TransferRequest::new(
                            ledger_canister_id,
                            transfer_arg,
                        ))
                    }
                    .await
                    .check_response(|response| {
                        if let Err(error) = response {
                            Err(format!("Minting failed: {error:?}"))
                        } else {
                            Ok(())
                        }
                    })
                    .with_workflow_position(0)
                    .push_outcome(&mut sale_outcome)
                    .result();
                }

                // 1. Call sns.new_sale_ticket
                if overall_result.is_ok() {
                    overall_result = {
                        let request =
                            sns_request_provider.new_sale_ticket(icp_amount, Some(sns_subaccount));
                        canister_agent.call_with_retries(
                            request,
                            SNS_ENDPOINT_RETRY_TIMEOUT,
                            SNS_ENDPOINT_RETRY_BACKOFF,
                            None,
                        )
                    }
                    .await
                    .check_response(|response| {
                        if let Err(error) = response.ticket() {
                            Err(format!("new_sale_ticket: {:?}", error.error_type()))
                        } else {
                            Ok(())
                        }
                    })
                    .with_workflow_position(1)
                    .push_outcome(&mut sale_outcome)
                    .result();
                }

                // 2. Call icp_ledger.transfer
                if overall_result.is_ok() {
                    overall_result = {
                        let sns_account = Account {
                            owner: sns_sale_canister_id.0,
                            subaccount: Some(sns_subaccount.0),
                        };
                        let transfer_arg = TransferArg {
                            from_subaccount: None,
                            to: sns_account,
                            fee: None,
                            created_at_time: None,
                            memo: None,
                            amount: Nat::from(icp_amount),
                        };
                        ledger_agent.call_and_parse(&Icrc1TransferRequest::new(
                            ledger_canister_id,
                            transfer_arg,
                        ))
                    }
                    .await
                    .check_response(|response| {
                        if let Err(error) = response {
                            Err(format!("ICP ledger transfer failed: {error:?}"))
                        } else {
                            Ok(())
                        }
                    })
                    .with_workflow_position(2)
                    .push_outcome(&mut sale_outcome)
                    .result();
                }

                // 3. Call sns.refresh_buyer_tokens
                if overall_result.is_ok() {
                    overall_result = {
                        let request = sns_request_provider.refresh_buyer_tokens(None);
                        canister_agent.call_with_retries(
                            request,
                            SNS_ENDPOINT_RETRY_TIMEOUT,
                            SNS_ENDPOINT_RETRY_BACKOFF,
                            None,
                        )
                    }
                    .await
                    .check_response(|_response| Ok(()))
                    .with_workflow_position(3)
                    .push_outcome(&mut sale_outcome)
                    .result();
                }

                // 4. Call sns.get_buyer_state
                if overall_result.is_ok() {
                    overall_result = {
                        let request = sns_request_provider
                            .get_buyer_state(Some(participant.principal_id), CallMode::Update);
                        canister_agent.call_with_retries(
                            request,
                            SNS_ENDPOINT_RETRY_TIMEOUT,
                            SNS_ENDPOINT_RETRY_BACKOFF,
                            None,
                        )
                    }
                    .await
                    .check_response(|response| {
                        let response_amount = response.buyer_state.unwrap().icp.unwrap().amount_e8s;
                        if response_amount >= icp_amount {
                            Ok(())
                        } else {
                            Err(format!("get_buyer_state: response ICP amount {response_amount:?} below the minimum amount {icp_amount:?}"))
                        }
                    })
                    .with_workflow_position(4)
                    .push_outcome(&mut sale_outcome)
                    .result();
                }

                // 5. Check that the ticket has been deleted via swap.get_open_ticket
                if overall_result.is_ok() {
                    overall_result = {
                        let request = sns_request_provider.get_open_ticket(CallMode::Update);
                        canister_agent.call_with_retries(
                            request,
                            SNS_ENDPOINT_RETRY_TIMEOUT,
                            SNS_ENDPOINT_RETRY_BACKOFF,
                            None,
                        )
                    }
                    .await
                    .check_response(|response| {
                        if response.ticket().unwrap().is_some() {
                            Err("get_open_ticket: ticket has not been deleted".to_string())
                        } else {
                            Ok(())
                        }
                    })
                    .with_workflow_position(5)
                    .push_outcome(&mut sale_outcome)
                    .result();
                }

                // Record e2e workflow metrics
                RequestOutcome::new(
                    overall_result,
                    "e2e_payment_flow".to_string(),
                    overall_start_time.elapsed(),
                    1,
                )
                .push_outcome(&mut sale_outcome);

                sale_outcome
            }
        }
    };
    let engine = Engine::new(log.clone(), future_generator, rps, duration)
        .increase_dispatch_timeout(SNS_ENDPOINT_RETRY_TIMEOUT);

    let metrics = {
        let rt = Builder::new_multi_thread()
            .worker_threads(16)
            .max_blocking_threads(16)
            .enable_all()
            .build()
            .unwrap();
        let aggr = LoadTestMetrics::new(log);
        let fun = LoadTestMetrics::aggregator_fn;
        rt.block_on(engine.execute(aggr, fun))
    }
    .unwrap();
    env.emit_report(format!("{metrics}"));
}

const SNS_ENDPOINT_RETRY_TIMEOUT: Duration = Duration::from_secs(5 * 60); // 5 minutes
const SNS_ENDPOINT_RETRY_BACKOFF: Duration = Duration::from_secs(2); // 2 seconds

pub fn workload_rps400_get_state_query(env: TestEnv) {
    let request = SnsRequestProvider::from_env(&env).get_state(CallMode::Query);
    generate_sns_workload(env, 400, WORKLOAD_GENERATION_DURATION, request);
}
pub fn workload_rps800_get_state_query(env: TestEnv) {
    let request = SnsRequestProvider::from_env(&env).get_state(CallMode::Query);
    generate_sns_workload(env, 800, WORKLOAD_GENERATION_DURATION, request);
}
pub fn workload_rps1200_get_state_query(env: TestEnv) {
    let request = SnsRequestProvider::from_env(&env).get_state(CallMode::Query);
    generate_sns_workload(env, 1_200, WORKLOAD_GENERATION_DURATION, request);
}

pub fn workload_rps400_get_state_update(env: TestEnv) {
    let request = SnsRequestProvider::from_env(&env).get_state(CallMode::Update);
    generate_sns_workload(env, 400, WORKLOAD_GENERATION_DURATION, request);
}
pub fn workload_rps800_get_state_update(env: TestEnv) {
    let request = SnsRequestProvider::from_env(&env).get_state(CallMode::Update);
    generate_sns_workload(env, 800, WORKLOAD_GENERATION_DURATION, request);
}
pub fn workload_rps1200_get_state_update(env: TestEnv) {
    let request = SnsRequestProvider::from_env(&env).get_state(CallMode::Update);
    generate_sns_workload(env, 1_200, WORKLOAD_GENERATION_DURATION, request);
}

pub fn workload_rps400_refresh_buyer_tokens(env: TestEnv) {
    let buyer = Some(*TEST_USER1_PRINCIPAL);
    let request = SnsRequestProvider::from_env(&env).refresh_buyer_tokens(buyer);
    generate_sns_workload(env, 400, WORKLOAD_GENERATION_DURATION, request);
}
pub fn workload_rps800_refresh_buyer_tokens(env: TestEnv) {
    let buyer = Some(*TEST_USER1_PRINCIPAL);
    let request = SnsRequestProvider::from_env(&env).refresh_buyer_tokens(buyer);
    generate_sns_workload(env, 800, WORKLOAD_GENERATION_DURATION, request);
}
pub fn workload_rps1200_refresh_buyer_tokens(env: TestEnv) {
    let buyer = Some(*TEST_USER1_PRINCIPAL);
    let request = SnsRequestProvider::from_env(&env).refresh_buyer_tokens(buyer);
    generate_sns_workload(env, 1_200, WORKLOAD_GENERATION_DURATION, request);
}

pub fn generate_sns_workload<T, R>(env: TestEnv, rps: usize, duration: Duration, request: R)
where
    T: Response,
    R: Request<T> + Clone + Sync + Send + 'static,
{
    let log = env.logger();

    // --- Generate workload ---
    let future_generator = {
        let agent = block_on(
            env.get_first_healthy_application_node_snapshot()
                .build_canister_agent(),
        );
        move |_idx| {
            let agent = agent.clone();
            let request = request.clone();
            async move { agent.call(&request).await.map(|_| ()).into_test_outcome() }
        }
    };
    let engine = Engine::new(log.clone(), future_generator, rps, duration)
        .increase_dispatch_timeout(REQUESTS_DISPATCH_EXTRA_TIMEOUT);

    let metrics = {
        let aggr = LoadTestMetrics::new(log);
        let fun = LoadTestMetrics::aggregator_fn;
        block_on(engine.execute(aggr, fun)).expect("Workload execution has failed.")
    };

    env.emit_report(format!("{metrics}"));
}
