/* tag::catalog[]
Title:: Specification compliance test

Goal:: Ensure that the replica implementation is compliant with the formal specification.

Runbook::
. Set up two subnets, each containing one node

Success:: The ic-ref-test binary does not return an error.

end::catalog[] */

use crate::canister_http::lib::{
    get_pem_content, get_universal_vm_activation_script, get_universal_vm_address, PemType,
};
use crate::driver::ic::{InternetComputer, Subnet};
use crate::driver::test_env::TestEnv;
use crate::driver::test_env_api::{
    HasPublicApiUrl, HasTopologySnapshot, IcNodeContainer, IcNodeSnapshot,
};
use crate::driver::universal_vm::{insert_file_to_config, UniversalVm, UniversalVms};
use ic_registry_subnet_features::SubnetFeatures;
use ic_registry_subnet_type::SubnetType;
use slog::{info, Logger};
use std::process::{Command, Stdio};

pub const UNIVERSAL_VM_NAME: &str = "httpbin";

const EXCLUDED: &[&str] = &[
    // to start with something that is always false
    "(1 == 0)",
    // tECDSA is not enabled in the test yet
    "$0 ~ /tECDSA/",
    // the replica does not yet check that the effective canister id is valid
    "$0 ~ /wrong effective canister id/",
    "$0 ~ /access denied two status to different canisters/",
    // the replica does not implement proofs of path non-existence
    "$0 ~ /non-existence proofs for non-existing request id/",
    "$0 ~ /module_hash of empty canister/",
    "$0 ~ /metadata.absent/",
];

pub fn config(env: TestEnv) {
    use crate::driver::test_env_api::{retry, secs};
    use crate::util::block_on;
    use hyper::client::connect::HttpConnector;
    use hyper::Client;
    use hyper_tls::HttpsConnector;
    use std::env;

    // Set up Universal VM with HTTP Bin testing service
    let activate_script = &get_universal_vm_activation_script()[..];
    let config_dir = env
        .single_activate_script_config_dir(UNIVERSAL_VM_NAME, activate_script)
        .unwrap();
    let _ = insert_file_to_config(
        config_dir.clone(),
        "cert.pem",
        get_pem_content(&env, &PemType::PemCert).as_bytes(),
    );
    let _ = insert_file_to_config(
        config_dir.clone(),
        "key.pem",
        get_pem_content(&env, &PemType::PemKey).as_bytes(),
    );
    env::set_var("SSL_CERT_FILE", config_dir.as_path().join("cert.pem"));
    env::remove_var("NIX_SSL_CERT_FILE");

    UniversalVm::new(String::from(UNIVERSAL_VM_NAME))
        .with_config_dir(config_dir)
        .start(&env)
        .expect("failed to set up universal VM");

    InternetComputer::new()
        .add_subnet(
            Subnet::new(SubnetType::System)
                .with_features(SubnetFeatures {
                    http_requests: true,
                    ..SubnetFeatures::default()
                })
                .add_nodes(1),
        )
        .add_subnet(
            Subnet::new(SubnetType::Application)
                .with_features(SubnetFeatures {
                    http_requests: true,
                    ..SubnetFeatures::default()
                })
                .add_nodes(1),
        )
        .setup_and_start(&env)
        .expect("failed to setup IC under test");
    env.topology_snapshot().subnets().for_each(|subnet| {
        subnet
            .nodes()
            .for_each(|node| node.await_status_is_healthy().unwrap())
    });

    let log = env.logger();
    let webserver_ipv6 = get_universal_vm_address(&env);
    let httpbin = format!("https://[{webserver_ipv6}]:20443");
    retry(log.clone(), secs(300), secs(10), || {
        block_on(async {
            let mut http_connector = HttpConnector::new();
            http_connector.enforce_http(false);
            let mut https_connector = HttpsConnector::new_with_connector(http_connector);
            https_connector.https_only(true);
            let client = Client::builder().build::<_, hyper::Body>(https_connector);

            let addr = format!("{}/ip", httpbin);
            let req = hyper::Request::builder()
                .method(hyper::Method::GET)
                .uri(addr)
                .body(hyper::Body::from(""))?;

            let resp = client.request(req).await?;

            let body_bytes = hyper::body::to_bytes(resp.into_body()).await?;
            let body = String::from_utf8(body_bytes.to_vec()).unwrap();

            info!(log, "response body from httpbin: {}", body);

            Ok(())
        })
    })
    .expect("Httpbin server should respond to incoming requests!");
}

pub fn test_system_subnet(env: TestEnv) {
    let log = env.logger();
    let topology_snapshot = env.topology_snapshot();
    let subnet = topology_snapshot.root_subnet();
    let node = subnet.nodes().next().unwrap();
    let webserver_ipv6 = get_universal_vm_address(&env);
    let httpbin = format!("https://[{webserver_ipv6}]:20443");
    with_endpoint(
        node,
        httpbin,
        log,
        [EXCLUDED.to_vec(), vec!["$0 ~ /only_application/"]].concat(),
    );
}

pub fn test_app_subnet(env: TestEnv) {
    let log = env.logger();
    let topology_snapshot = env.topology_snapshot();
    let subnet = topology_snapshot
        .subnets()
        .find(|s| s.subnet_type() == SubnetType::Application)
        .unwrap();
    let node = subnet.nodes().next().unwrap();
    let webserver_ipv6 = get_universal_vm_address(&env);
    let httpbin = format!("https://[{webserver_ipv6}]:20443");
    with_endpoint(
        node,
        httpbin,
        log,
        [EXCLUDED.to_vec(), vec!["$0 ~ /only_system/"]].concat(),
    );
}

pub fn with_endpoint(
    endpoint: IcNodeSnapshot,
    httpbin: String,
    log: Logger,
    excluded_tests: Vec<&str>,
) {
    let status = Command::new("ic-ref-test")
        .arg("-j8")
        .arg("--pattern")
        .arg(tests_to_pattern(excluded_tests))
        .arg("--endpoint")
        .arg(endpoint.get_public_url().to_string())
        .arg("--httpbin")
        .arg(&httpbin)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .expect("ic-ref-test binary crashed");
    info!(log, "{}", format!("Status of ic-ref-test: {:?}", &status));
    assert!(status.success());
}

fn tests_to_pattern(tests: Vec<&str>) -> String {
    format!("!({})", tests.join(" || "))
}
