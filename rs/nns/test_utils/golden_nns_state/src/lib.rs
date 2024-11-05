use ic_base_types::{PrincipalId, SubnetId};
use ic_pocket_ic_tests::{StateMachine, StateMachineBuilder};

use pocket_ic::common::rest::SubnetKind;
use std::{
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
};
use tempfile::TempDir;
// TODO: Add support for PocketIc.

pub fn new_state_machine_with_golden_fiduciary_state_or_panic() -> StateMachine {
    let fiduciary_subnet_id = SubnetId::new(
        PrincipalId::from_str("pzp6e-ekpqk-3c5x7-2h6so-njoeq-mt45d-h3h6c-q3mxf-vpeq5-fk5o7-yae")
            .unwrap(),
    );
    let setup_config = SetupConfig {
        archive_state_dir_name: "fiduciary_state",
        nonmainnet_features: true,
        scp_location: FIDUCIARY_STATE_SOURCE,
        subnet_id: fiduciary_subnet_id,
        subnet_kind: SubnetKind::Fiduciary,
    };
    new_state_machine_with_golden_state_or_panic(setup_config)
}

pub fn new_state_machine_with_golden_nns_state_or_panic() -> StateMachine {
    let nns_subnet_id = SubnetId::new(
        PrincipalId::from_str("tdb26-jop6k-aogll-7ltgs-eruif-6kk7m-qpktf-gdiqx-mxtrf-vb5e6-eqe")
            .unwrap(),
    );
    let setup_config = SetupConfig {
        archive_state_dir_name: "nns_state",
        nonmainnet_features: false,
        scp_location: NNS_STATE_SOURCE,
        subnet_id: nns_subnet_id,
        subnet_kind: SubnetKind::NNS,
    };
    new_state_machine_with_golden_state_or_panic(setup_config)
}

pub fn new_state_machine_with_golden_sns_state_or_panic() -> StateMachine {
    let sns_subnet_id = SubnetId::new(
        PrincipalId::from_str("x33ed-h457x-bsgyx-oqxqf-6pzwv-wkhzr-rm2j3-npodi-purzm-n66cg-gae")
            .unwrap(),
    );
    let setup_config = SetupConfig {
        archive_state_dir_name: "sns_state",
        nonmainnet_features: true,
        scp_location: SNS_STATE_SOURCE,
        subnet_id: sns_subnet_id,
        subnet_kind: SubnetKind::SNS,
    };
    new_state_machine_with_golden_state_or_panic(setup_config)
}

fn new_state_machine_with_golden_state_or_panic(setup_config: SetupConfig) -> StateMachine {
    let SetupConfig {
        archive_state_dir_name,
        nonmainnet_features,
        scp_location,
        subnet_id,
        subnet_kind,
    } = setup_config;
    let state_dir = maybe_download_golden_nns_state_or_panic(scp_location, archive_state_dir_name);
    let state_machine_builder = StateMachineBuilder::new()
        .with_current_time()
        .with_state_machine_state_dir(
            subnet_kind,
            subnet_id,
            state_dir.path(),
            nonmainnet_features,
        );

    println!("Building StateMachine...");
    let state_machine = state_machine_builder.build();
    println!("Done building StateMachine...");

    state_machine
}

/// A directory for storing the golden state which can be either a temporary directory or a cached
/// directory which can be used across multiple tests.
enum StateDir {
    // A temporary directory that will be deleted after the test is done.
    Temp(TempDir),
    // A directory that will be cached and reused across tests.
    Cache(PathBuf),
}

impl StateDir {
    fn path(&self) -> PathBuf {
        match self {
            Self::Temp(temp_dir) => temp_dir.path().to_path_buf(),
            Self::Cache(path) => path.clone(),
        }
    }
}

fn maybe_download_golden_nns_state_or_panic(
    scp_location: ScpLocation,
    archive_state_dir_name: &str,
) -> StateDir {
    let maybe_use_cached_state_dir = std::env::var_os("USE_CACHED_STATE_DIR");

    match maybe_use_cached_state_dir {
        Some(cached_state_dir) => {
            let destination = PathBuf::from(cached_state_dir).join(archive_state_dir_name);
            if !destination.exists() {
                std::fs::create_dir(&destination)
                    .unwrap_or_else(|_| panic!("Failed to create directory {destination:?}"));
                download_and_untar_golden_nns_state_or_panic(
                    scp_location,
                    archive_state_dir_name,
                    &destination,
                );
            }
            StateDir::Cache(destination)
        }
        None => {
            let state_dir = bazel_test_compatible_temp_dir_or_panic();
            download_and_untar_golden_nns_state_or_panic(
                scp_location,
                archive_state_dir_name,
                state_dir.path(),
            );
            StateDir::Temp(state_dir)
        }
    }
}

fn download_and_untar_golden_nns_state_or_panic(
    scp_location: ScpLocation,
    archive_state_dir_name: &str,
    destination: &Path,
) {
    let download_destination = bazel_test_compatible_temp_dir_or_panic();
    let download_destination = download_destination
        .path()
        .join(format!("{}.tar.zst", archive_state_dir_name));
    download_golden_nns_state_or_panic(scp_location, &download_destination);
    untar_state_archive_or_panic(&download_destination, destination, archive_state_dir_name);
}

// Privates

const FIDUCIARY_STATE_SOURCE: ScpLocation = ScpLocation {
    user: "dev",
    host: "zh1-pyr07.zh1.dfinity.network",
    path: "/home/dev/fiduciary_state.tar.zst",
};

const NNS_STATE_SOURCE: ScpLocation = ScpLocation {
    user: "dev",
    host: "zh1-pyr07.zh1.dfinity.network",
    path: "/home/dev/nns_state.tar.zst",
};

const SNS_STATE_SOURCE: ScpLocation = ScpLocation {
    user: "dev",
    host: "zh1-pyr07.zh1.dfinity.network",
    path: "/home/dev/sns_state.tar.zst",
};

/// A place that you can download from or upload to using the `scp` command.
#[derive(Debug)]
struct ScpLocation {
    user: &'static str,
    host: &'static str,
    path: &'static str,
}

impl ScpLocation {
    pub fn to_argument(&self) -> String {
        let Self { user, host, path } = self;

        format!("{}@{}:{}", user, host, path)
    }
}

struct SetupConfig {
    archive_state_dir_name: &'static str,
    nonmainnet_features: bool,
    scp_location: ScpLocation,
    subnet_id: SubnetId,
    subnet_kind: SubnetKind,
}

fn download_golden_nns_state_or_panic(scp_location: ScpLocation, destination: &Path) {
    let source = scp_location.to_argument();
    println!("Downloading {} to {:?} ...", source, destination,);

    // Actually download.
    let scp_out = Command::new("scp")
        .arg("-oUserKnownHostsFile=/dev/null")
        .arg("-oStrictHostKeyChecking=no")
        .arg("-v")
        .arg(source.clone())
        .arg(destination)
        .output()
        .unwrap_or_else(|err| panic!("Could not scp from {:?} because: {:?}!", scp_location, err));

    // Inspect result.
    if !scp_out.status.success() {
        panic!("Could not scp from {}\n{:#?}", source, scp_out,);
    }

    let size = std::fs::metadata(destination)
        .map(|metadata| {
            let len = metadata.len() as f64;
            let len = len / (1 << 30) as f64;
            format!("{:.2} GiB", len)
        })
        .unwrap_or_else(|_err| "???".to_string());

    let destination = destination.to_string_lossy();
    println!("Downloaded {} to {}. size = {}", source, destination, size);
}

fn untar_state_archive_or_panic(source: &Path, destination: &Path, state_dir: &str) {
    println!(
        "Unpacking {} from {:?} to {:?}...",
        state_dir, source, destination
    );

    // TODO: Mathias reports having problems with this (or something similar) on Mac.
    let unpack_destination = bazel_test_compatible_temp_dir_or_panic();
    let unpack_destination = unpack_destination
        .path()
        .to_str()
        .expect("Was trying to convert a Path to a string.");
    let tar_out = Command::new("tar")
        .arg("--extract")
        .arg("--file")
        .arg(source)
        .arg("--directory")
        .arg(unpack_destination)
        .output()
        .unwrap_or_else(|err| panic!("Could not unpack {:?}: {}", source, err));

    if !tar_out.status.success() {
        panic!("Could not unpack {:?}\n{:#?}", source, tar_out);
    }

    // Move $UNTAR_DESTINATION/nns_state/ic_state to final output dir path, StateMachine's so-called
    // state_dir.
    std::fs::rename(
        format!("{}/{}/ic_state", unpack_destination, state_dir),
        destination,
    )
    .unwrap();

    println!("Unpacked {:?} to {:?}", source, destination);
}

/// If available, uses the `TEST_TMPDIR` environment variable, which is set by
/// `bazel test`, and points to where you are allowed to write to disk.
/// Otherwise, this just falls back on vanilla TempDir::new.
fn bazel_test_compatible_temp_dir_or_panic() -> TempDir {
    match std::env::var("TEST_TMPDIR") {
        Ok(dir) => TempDir::new_in(dir).unwrap(),
        Err(_err) => TempDir::new().unwrap(),
    }
}
