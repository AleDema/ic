use candid::{CandidType, Deserialize};
use dfn_core::api::{call, CanisterId};
use rust_decimal::Decimal;
use serde::Serialize;

use std::convert::TryInto;
use std::fmt;
use std::fmt::Formatter;

use ic_base_types::PrincipalId;
use ic_ic00_types::{CanisterIdRecord, CanisterStatusResultV2, IC_00};

pub mod ledger;
pub mod stable_mem_utils;

// 10^8
pub const E8: u64 = 100_000_000;

pub const SECONDS_PER_DAY: u64 = 24 * 60 * 60;

// Useful as a piece of realistic test data.
pub const START_OF_2022_TIMESTAMP_SECONDS: u64 = 1641016800;

#[macro_export]
macro_rules! assert_is_ok {
    ($result: expr) => {
        let r = $result;
        assert!(
            r.is_ok(),
            "result ({}) = {:#?}, not Ok",
            stringify!($result),
            r
        );
    };
}

#[macro_export]
macro_rules! assert_is_err {
    ($result: expr) => {
        let r = $result;
        assert!(
            r.is_err(),
            "result ({}) = {:#?}, not Err",
            stringify!($result),
            r
        );
    };
}

pub fn i2d(i: u64) -> Decimal {
    // Convert to i64.
    let i = i
        .try_into()
        .unwrap_or_else(|err| panic!("{} does not fit into i64: {:#?}", i, err));

    Decimal::new(i, 0)
}

/// A general purpose error indicating something went wrong.
#[derive(Default)]
pub struct NervousSystemError {
    pub error_message: String,
}

impl NervousSystemError {
    pub fn new() -> Self {
        NervousSystemError {
            ..Default::default()
        }
    }

    pub fn new_with_message(message: impl ToString) -> Self {
        NervousSystemError {
            error_message: message.to_string(),
        }
    }
}

impl fmt::Display for NervousSystemError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.error_message)
    }
}

impl fmt::Debug for NervousSystemError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error_message)
    }
}

/// Description of a change to the authz of a specific method on a specific
/// canister that must happen for a given canister change/add/remove
/// to be viable
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub struct MethodAuthzChange {
    pub canister: CanisterId,
    pub method_name: String,
    pub principal: Option<PrincipalId>,
    pub operation: AuthzChangeOp,
}

/// The operation to execute. Varible names in comments refer to the fields
/// of AuthzChange.
#[derive(CandidType, Serialize, Deserialize, Clone, Debug)]
pub enum AuthzChangeOp {
    /// 'canister' must add a principal to the authorized list of 'method_name'.
    /// If 'add_self' is true, the canister_id to be authorized is the canister
    /// being added/changed, if it's false, 'principal' is used instead, which
    /// must be Some in that case..
    Authorize { add_self: bool },
    /// 'canister' must remove 'principal' from the authorized list of
    /// 'method_name'. 'principal' must always be Some.
    Deauthorize,
}

/// Return the status of the given canister. The caller must control the given canister.
pub async fn get_canister_status(
    canister_id: PrincipalId,
) -> Result<CanisterStatusResultV2, (Option<i32>, String)> {
    let canister_id_record: CanisterIdRecord = CanisterId::new(canister_id).unwrap().into();

    call(
        IC_00,
        "canister_status",
        dfn_candid::candid,
        (canister_id_record,),
    )
    .await
}
