//! State modifications that should end up in the event log.

use super::{
    eventlog::Event, CkBtcMinterState, FinalizedBtcRetrieval, FinalizedStatus, RetrieveBtcRequest,
    SubmittedBtcTransaction, UtxoCheckStatus,
};
use crate::storage::record_event;
use candid::Principal;
use ic_btc_types::Utxo;
use icrc_ledger_types::Account;

pub fn accept_retrieve_btc_request(state: &mut CkBtcMinterState, request: RetrieveBtcRequest) {
    record_event(&Event::AcceptedRetrieveBtcRequest(request.clone()));
    state.pending_retrieve_btc_requests.push(request);
}

pub fn add_utxos(
    state: &mut CkBtcMinterState,
    mint_txid: Option<u64>,
    account: Account,
    utxos: Vec<Utxo>,
) {
    record_event(&Event::ReceivedUtxos {
        mint_txid,
        to_account: account,
        utxos: utxos.clone(),
    });

    state.add_utxos(account, utxos);
}

pub fn remove_retrieve_btc_request(state: &mut CkBtcMinterState, request: RetrieveBtcRequest) {
    record_event(&Event::RemovedRetrieveBtcRequest {
        block_index: request.block_index,
    });

    state.push_finalized_request(FinalizedBtcRetrieval {
        request,
        state: FinalizedStatus::AmountTooLow,
    });
}

pub fn sent_transaction(state: &mut CkBtcMinterState, tx: SubmittedBtcTransaction) {
    record_event(&Event::SentBtcTransaction {
        request_block_indices: tx.requests.iter().map(|r| r.block_index).collect(),
        txid: tx.txid,
        utxos: tx.used_utxos.clone(),
        change_output: tx.change_output.clone(),
        submitted_at: tx.submitted_at,
    });

    state.push_submitted_transaction(tx);
}

pub fn confirm_transaction(state: &mut CkBtcMinterState, txid: &[u8; 32]) {
    record_event(&Event::ConfirmedBtcTransaction { txid: *txid });
    state.finalize_transaction(txid);
}

pub fn mark_utxo_checked(
    state: &mut CkBtcMinterState,
    utxo: &Utxo,
    uuid: String,
    status: UtxoCheckStatus,
    kyt_provider: Principal,
) {
    record_event(&Event::CheckedUtxo {
        utxo: utxo.clone(),
        uuid: uuid.clone(),
        clean: status.is_clean(),
        kyt_provider,
    });
    state.mark_utxo_checked(utxo.clone(), uuid, status, kyt_provider);
}

pub fn ignore_utxo(state: &mut CkBtcMinterState, utxo: Utxo) {
    record_event(&Event::IgnoredUtxo { utxo: utxo.clone() });
    state.ignore_utxo(utxo);
}
