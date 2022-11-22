use candid::candid_method;
use ic_cdk_macros::{heartbeat, init, post_upgrade, pre_upgrade, query, update};
use ic_ckbtc_minter::lifecycle::{self, init::InitArgs};
use ic_ckbtc_minter::metrics::encode_metrics;
use ic_ckbtc_minter::queries::RetrieveBtcStatusRequest;
use ic_ckbtc_minter::state::{read_state, RetrieveBtcStatus};
use ic_ckbtc_minter::updates::retrieve_btc::{RetrieveBtcArgs, RetrieveBtcError, RetrieveBtcOk};
use ic_ckbtc_minter::updates::{
    self,
    get_btc_address::GetBtcAddressArgs,
    get_withdrawal_account::GetWithdrawalAccountResult,
    update_balance::{UpdateBalanceArgs, UpdateBalanceError, UpdateBalanceResult},
};

#[init]
fn init(args: InitArgs) {
    lifecycle::init::init(args)
}

#[heartbeat]
async fn heartbeat() {
    ic_ckbtc_minter::heartbeat().await;
}

#[pre_upgrade]
fn pre_upgrade() {
    lifecycle::upgrade::pre_upgrade()
}

#[post_upgrade]
fn post_upgrade() {
    lifecycle::upgrade::post_upgrade()
}

#[candid_method(update)]
#[update]
async fn get_btc_address(args: GetBtcAddressArgs) -> String {
    updates::get_btc_address::get_btc_address(args).await
}

#[candid_method(update)]
#[update]
async fn get_withdrawal_account() -> GetWithdrawalAccountResult {
    updates::get_withdrawal_account::get_withdrawal_account().await
}

#[candid_method(update)]
#[update]
async fn retrieve_btc(args: RetrieveBtcArgs) -> Result<RetrieveBtcOk, RetrieveBtcError> {
    updates::retrieve_btc::retrieve_btc(args).await
}

#[candid_method(query)]
#[query]
fn retrieve_btc_status(req: RetrieveBtcStatusRequest) -> RetrieveBtcStatus {
    read_state(|s| s.retrieve_btc_status(req.block_index))
}

#[candid_method(update)]
#[update]
async fn update_balance(
    args: UpdateBalanceArgs,
) -> Result<UpdateBalanceResult, UpdateBalanceError> {
    updates::update_balance::update_balance(args).await
}

#[export_name = "canister_query http_request"]
fn http_request() {
    dfn_http_metrics::serve_metrics(encode_metrics);
}

#[query]
fn __get_candid_interface_tmp_hack() -> &'static str {
    include_str!(env!("CKBTC_MINTER_DID_PATH"))
}

fn main() {}

/// Checks the real candid interface against the one declared in the did file
#[test]
fn check_candid_interface_compatibility() {
    fn source_to_str(source: &candid::utils::CandidSource) -> String {
        match source {
            candid::utils::CandidSource::File(f) => {
                std::fs::read_to_string(f).unwrap_or_else(|_| "".to_string())
            }
            candid::utils::CandidSource::Text(t) => t.to_string(),
        }
    }

    fn check_service_compatible(
        new_name: &str,
        new: candid::utils::CandidSource,
        old_name: &str,
        old: candid::utils::CandidSource,
    ) {
        let new_str = source_to_str(&new);
        let old_str = source_to_str(&old);
        match candid::utils::service_compatible(new, old) {
            Ok(_) => {}
            Err(e) => {
                eprintln!(
                    "{} is not compatible with {}!\n\n\
            {}:\n\
            {}\n\n\
            {}:\n\
            {}\n",
                    new_name, old_name, new_name, new_str, old_name, old_str
                );
                panic!("{:?}", e);
            }
        }
    }

    candid::export_service!();

    let new_interface = __export_service();

    // check the public interface against the actual one
    let old_interface = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("ckbtc_minter.did");

    check_service_compatible(
        "actual ledger candid interface",
        candid::utils::CandidSource::Text(&new_interface),
        "declared candid interface in ckbtc_minter.did file",
        candid::utils::CandidSource::File(old_interface.as_path()),
    );
}
