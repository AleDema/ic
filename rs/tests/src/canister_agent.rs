use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use anyhow::bail;
use async_trait::async_trait;
use ic_agent::{Agent, Identity};

use crate::{
    canister_api::{Request, Response},
    driver::test_env_api::{retry_async, HasPublicApiUrl, IcNodeSnapshot},
    generic_workload_engine::metrics::RequestOutcome,
    util::{assert_create_agent, assert_create_agent_with_identity, delay},
};

/// An agent that is well-suited for interacting with arbitrary IC canisters.
#[derive(Clone)]
pub struct CanisterAgent {
    pub agent: Agent,
}

#[async_trait]
pub trait HasCanisterAgentCapability: HasPublicApiUrl + Send + Sync {
    async fn build_canister_agent(&self) -> CanisterAgent;
    async fn build_canister_agent_with_identity(
        &self,
        identity: impl Identity + Clone + 'static,
    ) -> CanisterAgent;
}

#[async_trait]
impl HasCanisterAgentCapability for IcNodeSnapshot {
    async fn build_canister_agent(&self) -> CanisterAgent {
        let agent = assert_create_agent(self.get_public_url().as_str()).await;
        CanisterAgent { agent }
    }

    async fn build_canister_agent_with_identity(
        &self,
        identity: impl Identity + Clone + 'static,
    ) -> CanisterAgent {
        let agent =
            assert_create_agent_with_identity(self.get_public_url().as_str(), identity).await;
        CanisterAgent { agent }
    }
}

impl CanisterAgent {
    pub async fn from_boundary_node_url(bn_url: &str) -> Self {
        Self {
            agent: assert_create_agent(bn_url).await,
        }
    }

    pub fn get(&self) -> Agent {
        self.agent.clone()
    }

    pub async fn call<T>(
        &self,
        request: &(dyn Request<T> + Send + Sync),
    ) -> RequestOutcome<Vec<u8>, String>
    where
        T: Response,
    {
        let start_time = Instant::now();
        let result = (if request.is_query() {
            self.agent
                .query(&request.canister_id(), request.method_name())
                .with_arg(request.payload())
                .call()
                .await
        } else {
            self.agent
                .update(&request.canister_id(), request.method_name())
                .with_arg(request.payload())
                .call_and_wait(delay())
                .await
        })
        .map_err(|e| format!("{e:?}"));
        RequestOutcome::new(
            result,
            // TODO: incorporate canister IDs into labels to avoid collisions if two canisters have endpoints with the same name
            request.method_name(),
            start_time.elapsed(),
            1,
        )
    }

    pub async fn call_and_parse<T>(
        &self,
        request: &(dyn Request<T> + Send + Sync),
    ) -> RequestOutcome<T, String>
    where
        T: Response + Clone,
    {
        let raw_outcome = self.call(request).await;
        RequestOutcome::new(
            raw_outcome.result().map(|r| {
                request
                    .parse_response(r.as_slice())
                    .expect("failed to decode")
            }),
            format!("{}+parse", request.method_name()),
            raw_outcome.duration,
            raw_outcome.attempts,
        )
    }

    pub async fn call_with_retries<T, R>(
        &self,
        request: R,
        timeout: Duration,
        backoff_delay: Duration,
        expected_outcome: Option<&(dyn Fn(T) -> bool + Sync + Send)>,
    ) -> RequestOutcome<T, String>
    where
        T: Response + Clone,
        R: Request<T> + Clone + Sync + Send + 'static,
    {
        let log = slog::Logger::root(slog::Discard, slog::o!());
        let label = request.method_name();
        let start_time = Instant::now();
        let attempts = Arc::new(AtomicUsize::new(0));
        let result = retry_async(&log, timeout, backoff_delay, {
            let attempts = attempts.clone();
            let request = request.clone();
            move || {
                attempts.fetch_add(1, Ordering::Relaxed);
                let request = request.clone();
                async move {
                    let request = request.clone();
                    match self.call_and_parse(&request).await.result() {
                        Ok(result) => match expected_outcome {
                            Some(predicate) => {
                                if predicate(result.clone()) {
                                    Ok(result)
                                } else {
                                    bail!("Unexpected outcome")
                                }
                            }
                            None => Ok(result),
                        },
                        Err(error) => {
                            bail!(error)
                        }
                    }
                }
            }
        })
        .await
        .map_err(|e| format!("{e:?}"));
        RequestOutcome::new(
            result,
            format!("{}+retries", label),
            start_time.elapsed(),
            attempts.load(Ordering::Relaxed),
        )
    }
}
