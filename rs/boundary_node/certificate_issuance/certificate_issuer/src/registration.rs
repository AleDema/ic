use std::{sync::Arc, time::Duration};

use anyhow::{anyhow, Context};
use async_trait::async_trait;
use candid::{Decode, Encode, Principal};
use certificate_orchestrator_interface as ifc;
use garcon::Delay;
use ic_agent::Agent;
use serde::{Deserialize, Serialize};

use crate::work::ProcessError;

pub type Id = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum State {
    Failed(String),
    PendingOrder,
    PendingChallengeResponse,
    PendingAcmeApproval,
    Available,
}

impl ToString for State {
    fn to_string(&self) -> String {
        serde_json::ser::to_string(self).unwrap_or_else(|_| "N/A".into())
    }
}

impl From<ProcessError> for State {
    fn from(e: ProcessError) -> Self {
        match e {
            ProcessError::AwaitingDnsPropogation => State::PendingChallengeResponse,
            ProcessError::AwaitingAcmeOrderReady => State::PendingAcmeApproval,
            ProcessError::UnexpectedError(_) => State::Failed(e.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Registration {
    pub name: String,
    pub canister: Principal,
    pub state: State,
}

#[derive(Debug, thiserror::Error)]
pub enum CreateError {
    #[error("Registration '{0}' already exists")]
    Duplicate(Id),
    #[error(transparent)]
    UnexpectedError(#[from] anyhow::Error),
}

#[async_trait]
pub trait Create: Send + Sync {
    async fn create(&self, name: &str, canister: &Principal) -> Result<Id, CreateError>;
}

#[derive(Debug, thiserror::Error)]
pub enum UpdateError {
    #[error("Not found")]
    NotFound,
    #[error(transparent)]
    UnexpectedError(#[from] anyhow::Error),
}

#[async_trait]
pub trait Update: Send + Sync {
    async fn update(&self, id: &Id, state: &State) -> Result<(), UpdateError>;
}

#[derive(Debug, thiserror::Error)]
pub enum GetError {
    #[error("Not found")]
    NotFound,
    #[error(transparent)]
    UnexpectedError(#[from] anyhow::Error),
}

#[async_trait]
pub trait Get: Send + Sync {
    async fn get(&self, id: &Id) -> Result<Registration, GetError>;
}

impl From<ifc::State> for State {
    fn from(s: ifc::State) -> Self {
        match s {
            ifc::State::Failed(err) => State::Failed(err),
            ifc::State::PendingOrder => State::PendingOrder,
            ifc::State::PendingChallengeResponse => State::PendingChallengeResponse,
            ifc::State::PendingAcmeApproval => State::PendingAcmeApproval,
            ifc::State::Available => State::Available,
        }
    }
}

impl From<State> for ifc::State {
    fn from(s: State) -> Self {
        match s {
            State::Failed(err) => ifc::State::Failed(err),
            State::PendingOrder => ifc::State::PendingOrder,
            State::PendingChallengeResponse => ifc::State::PendingChallengeResponse,
            State::PendingAcmeApproval => ifc::State::PendingAcmeApproval,
            State::Available => ifc::State::Available,
        }
    }
}

impl From<ifc::Registration> for Registration {
    fn from(reg: ifc::Registration) -> Self {
        Registration {
            name: reg.name.into(),
            canister: reg.canister,
            state: reg.state.into(),
        }
    }
}

pub struct CanisterGetter(pub Arc<Agent>, pub Principal);

#[async_trait]
impl Get for CanisterGetter {
    async fn get(&self, id: &Id) -> Result<Registration, GetError> {
        use ifc::{GetRegistrationError as Error, GetRegistrationResponse as Response};

        let args = Encode!(id).context("failed to encode arg")?;

        let resp = self
            .0
            .query(&self.1, "getRegistration")
            .with_arg(args)
            .call()
            .await
            .context("failed to query canister")?;

        let resp = Decode!(&resp, Response).context("failed to decode canister response")?;

        match resp {
            Response::Ok(reg) => Ok(reg.into()),
            Response::Err(err) => Err(match err {
                Error::NotFound => GetError::NotFound,
                Error::Unauthorized => GetError::UnexpectedError(anyhow!("unauthorized")),
                Error::UnexpectedError(err) => GetError::UnexpectedError(anyhow!(err)),
            }),
        }
    }
}

pub struct CanisterCreator(pub Arc<Agent>, pub Principal);

#[async_trait]
impl Create for CanisterCreator {
    async fn create(&self, name: &str, canister: &Principal) -> Result<Id, CreateError> {
        use ifc::{CreateRegistrationError as Error, CreateRegistrationResponse as Response};

        let waiter = Delay::builder()
            .throttle(Duration::from_millis(500))
            .timeout(Duration::from_millis(3000))
            .build();

        let args = Encode!(&name.to_string(), canister).context("failed to encode arg")?;

        let resp = self
            .0
            .update(&self.1, "createRegistration")
            .with_arg(args)
            .call_and_wait(waiter)
            .await
            .context("failed to query canister")?;

        let resp = Decode!(&resp, Response).context("failed to decode canister response")?;

        match resp {
            Response::Ok(id) => Ok(id),
            Response::Err(err) => Err(match err {
                Error::Duplicate(id) => CreateError::Duplicate(id),
                Error::NameError(err) => CreateError::UnexpectedError(anyhow!(err)),
                Error::Unauthorized => CreateError::UnexpectedError(anyhow!("unauthorized")),
                Error::UnexpectedError(err) => CreateError::UnexpectedError(anyhow!(err)),
            }),
        }
    }
}

pub struct CanisterUpdater(pub Arc<Agent>, pub Principal);

#[async_trait]
impl Update for CanisterUpdater {
    async fn update(&self, id: &Id, state: &State) -> Result<(), UpdateError> {
        use ifc::{UpdateRegistrationError as Error, UpdateRegistrationResponse as Response};

        let waiter = Delay::builder()
            .throttle(Duration::from_millis(500))
            .timeout(Duration::from_millis(3000))
            .build();

        let state: ifc::State = state.to_owned().into();
        let args = Encode!(id, &state).context("failed to encode arg")?;

        let resp = self
            .0
            .update(&self.1, "updateRegistration")
            .with_arg(args)
            .call_and_wait(waiter)
            .await
            .context("failed to query canister")?;

        let resp = Decode!(&resp, Response).context("failed to decode canister response")?;

        match resp {
            Response::Ok(()) => Ok(()),
            Response::Err(err) => Err(match err {
                Error::NotFound => UpdateError::NotFound,
                Error::Unauthorized => UpdateError::UnexpectedError(anyhow!("unauthorized")),
                Error::UnexpectedError(err) => UpdateError::UnexpectedError(anyhow!(err)),
            }),
        }
    }
}
