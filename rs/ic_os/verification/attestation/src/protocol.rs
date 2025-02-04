use crate::attestation::SevAttestationReport;
use candid::CandidType;
use std::fmt::{Debug, Display};

pub use crate::error::{VerificationError, VerificationErrorDetail};

#[derive(CandidType, candid::Deserialize)]
pub struct GenerateAttestationTokenChallenge {
    pub nonce: Vec<u8>,
}

#[derive(CandidType, candid::Deserialize)]
pub struct InitiateGenerateAttestationTokenRequest {
    pub tls_public_key: Vec<u8>,
}

#[derive(CandidType, candid::Deserialize)]
pub struct InitiateGenerateAttestationTokenResponse {
    pub challenge: GenerateAttestationTokenChallenge,
}

#[derive(CandidType, candid::Deserialize)]
pub struct GenerateAttestationTokenRequest {
    pub tls_public_key: Vec<u8>,
    pub nonce: Vec<u8>,
    pub sev_attestation_report: SevAttestationReport,
}

#[derive(CandidType, candid::Deserialize)]
pub struct GenerateAttestationTokenResponse {}

#[derive(CandidType, candid::Deserialize)]
pub struct FetchAttestationTokenRequest {
    pub tls_public_key: Vec<u8>,
}

#[derive(CandidType, candid::Deserialize)]
pub struct FetchAttestationTokenResponse {
    pub attestation_token: Vec<u8>,
}
