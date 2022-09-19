use ic_cdk_macros::{heartbeat, init, update};
use ic_icrc1_index::{GetAccountTransactionsArgs, GetTransactionsResult, InitArgs};

fn main() {}

#[init]
fn init(args: InitArgs) {
    ic_icrc1_index::init(args);
}

#[heartbeat]
async fn heartbeat() {
    ic_icrc1_index::heartbeat().await;
}

#[update]
async fn get_account_transactions(args: GetAccountTransactionsArgs) -> GetTransactionsResult {
    ic_icrc1_index::get_account_transactions(args).await
}
