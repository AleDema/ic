#[cfg(test)]
mod tests {
    use ic_base_types::{NodeId, RegistryVersion};
    use ic_config::transport::TransportConfig;
    use ic_crypto::utils::TempCryptoComponent;
    use ic_crypto_tls_interfaces::{TlsClientHandshakeError, TlsHandshake};
    use ic_crypto_tls_interfaces_mocks::MockTlsHandshake;
    use ic_interfaces_transport::{
        FlowTag, Transport, TransportError, TransportEvent, TransportEventHandler, TransportPayload,
    };
    use ic_logger::ReplicaLogger;
    use ic_metrics::MetricsRegistry;
    use ic_registry_client_fake::FakeRegistryClient;
    use ic_registry_keys::make_crypto_tls_cert_key;
    use ic_registry_proto_data_provider::ProtoRegistryDataProvider;
    use ic_test_utilities_logger::with_test_replica_logger;
    use ic_transport::transport::create_transport;
    use ic_types_test_utils::ids::{NODE_1, NODE_2};
    use std::convert::Infallible;
    use std::net::SocketAddr;
    use std::str::FromStr;
    use std::sync::Arc;
    use tokio::net::{TcpSocket, TcpStream};
    use tokio::sync::mpsc::{channel, Sender};
    use tokio::sync::Notify;
    use tokio::time::Duration;
    use tower::{util::BoxCloneService, Service, ServiceExt};
    use tower_test::mock::Handle;

    const NODE_ID_1: NodeId = NODE_1;
    const NODE_ID_2: NodeId = NODE_2;
    const REG_V1: RegistryVersion = RegistryVersion::new(1);
    const FLOW_TAG: u32 = 1234;

    fn setup_test_peer<F>(
        log: ReplicaLogger,
        rt_handle: tokio::runtime::Handle,
        node_id: NodeId,
        port: u16,
        registry_version: RegistryVersion,
        registry_and_data: &mut RegistryAndDataProvider,
        mut crypto_factory: F,
    ) -> (Arc<dyn Transport>, Handle<TransportEvent, ()>, SocketAddr)
    where
        F: FnMut(&mut RegistryAndDataProvider, NodeId) -> Arc<dyn TlsHandshake + Send + Sync>,
    {
        let crypto = crypto_factory(registry_and_data, node_id);
        let config = TransportConfig {
            node_ip: "0.0.0.0".to_string(),
            legacy_flow_tag: FLOW_TAG,
            listening_port: port,
            send_queue_size: 10,
        };
        let peer = create_transport(
            node_id,
            config,
            registry_version,
            MetricsRegistry::new(),
            crypto,
            rt_handle,
            log,
        );
        let addr = SocketAddr::from_str(&format!("127.0.0.1:{}", port)).unwrap();
        let (event_handler, mock_handle) = create_mock_event_handler();
        peer.set_event_handler(event_handler);
        (peer, mock_handle, addr)
    }

    #[test]
    fn test_start_connection_between_two_peers() {
        with_test_replica_logger(|logger| {
            let registry_version = REG_V1;

            let rt = tokio::runtime::Runtime::new().unwrap();

            let (connected_1, mut done_1) = channel(1);
            let event_handler_1 = setup_peer_up_ack_event_handler(rt.handle().clone(), connected_1);

            let (connected_2, mut done_2) = channel(1);
            let event_handler_2 = setup_peer_up_ack_event_handler(rt.handle().clone(), connected_2);

            let (_control_plane_1, _control_plane_2) = start_connection_between_two_peers(
                rt.handle().clone(),
                logger,
                registry_version,
                10,
                event_handler_1,
                event_handler_2,
            );

            assert_eq!(done_1.blocking_recv(), Some(true));
            assert_eq!(done_2.blocking_recv(), Some(true));
        });
    }

    /*
    Verifies that transport suffers "head of line problem" when peer is slow to consume messages.
    - Peer A sends Peer B message, which will work fine.
    - Then, B's event handler blocks to prevent B from reading additional messages.
    - A sends a few more messages, but at this point queue will be full.
    - Finally, we unblock B's event handler, and confirm all in-flight messages are delivered.
    */
    #[test]
    fn head_of_line_test() {
        let registry_version = REG_V1;
        with_test_replica_logger(|logger| {
            // Setup registry and crypto component
            let rt = tokio::runtime::Runtime::new().unwrap();

            let (connected_1, _done_1) = channel(5);
            let event_handler_1 = setup_peer_up_ack_event_handler(rt.handle().clone(), connected_1);

            let (connected_2, mut done_2) = channel(5);

            let notify = Arc::new(Notify::new());
            let listener = notify.clone();

            let (hol_event_handler, mut hol_handle) = create_mock_event_handler();
            let blocking_msg = TransportPayload(vec![0xa; 1000000]);
            let normal_msg = TransportPayload(vec![0xb; 1000000]);

            let blocking_msg_copy = blocking_msg.clone();
            // Create event handler that blocks on message
            rt.spawn(async move {
                loop {
                    let (event, rsp) = hol_handle.next_request().await.unwrap();
                    match event {
                        TransportEvent::Message(msg) => {
                            connected_2.send(true).await.expect("Channel full");
                            // This will block the read task
                            if msg.payload == blocking_msg_copy {
                                listener.notified().await;
                            }
                        }
                        TransportEvent::PeerUp(_) => {}
                        TransportEvent::PeerDown(_) => {}
                    };
                    rsp.send_response(());
                }
            });

            let (client, _server) = start_connection_between_two_peers(
                rt.handle().clone(),
                logger,
                registry_version,
                1,
                event_handler_1,
                hol_event_handler,
            );

            let flow_tag = FlowTag::from(FLOW_TAG);

            // Send message from A -> B
            let res = client.send(&NODE_ID_2, flow_tag, blocking_msg);
            assert_eq!(res, Ok(()));
            assert_eq!(done_2.blocking_recv(), Some(true));

            // Send more messages from A->B until TCP Queue is full
            // Then, A's send queue should be blocked from dequeuing, triggering error
            let mut messages_sent = 0;
            loop {
                let _temp = normal_msg.clone();
                if let Err(TransportError::SendQueueFull(ref _temp)) =
                    client.send(&NODE_ID_2, flow_tag, normal_msg.clone())
                {
                    break;
                }
                messages_sent += 1;
                std::thread::sleep(Duration::from_millis(10));
            }
            let res2 = client.send(&NODE_ID_2, flow_tag, normal_msg.clone());
            assert_eq!(res2, Err(TransportError::SendQueueFull(normal_msg)));

            // Unblock event handler and confirm in-flight messages are received.
            notify.notify_one();

            for _ in 1..=messages_sent {
                assert_eq!(done_2.blocking_recv(), Some(true));
            }
        });
    }

    /*
    Establish connection with 2 peers, A and B.  Send message from A->B and B->A and confirm both are received
    */
    #[test]
    fn test_basic_message_send() {
        let registry_version = REG_V1;
        with_test_replica_logger(|logger| {
            let rt = tokio::runtime::Runtime::new().unwrap();

            let (peer_a_sender, mut peer_a_receiver) = channel(1);
            let peer_a_event_handler =
                setup_message_ack_event_handler(rt.handle().clone(), peer_a_sender);

            let (peer_b_sender, mut peer_b_receiver) = channel(1);
            let peer_b_event_handler =
                setup_message_ack_event_handler(rt.handle().clone(), peer_b_sender);

            let (peer_a, peer_b) = start_connection_between_two_peers(
                rt.handle().clone(),
                logger,
                registry_version,
                1,
                peer_a_event_handler,
                peer_b_event_handler,
            );

            let msg_1 = TransportPayload(vec![0xa; 1000000]);
            let msg_2 = TransportPayload(vec![0xb; 1000000]);
            let flow_tag = FlowTag::from(FLOW_TAG);

            // A sends message to B
            let res = peer_a.send(&NODE_ID_2, flow_tag, msg_1.clone());
            assert_eq!(res, Ok(()));
            assert_eq!(peer_b_receiver.blocking_recv(), Some(msg_1));

            // B sends message to A
            let res2 = peer_b.send(&NODE_ID_1, flow_tag, msg_2.clone());
            assert_eq!(res2, Ok(()));
            assert_eq!(peer_a_receiver.blocking_recv(), Some(msg_2));
        });
    }

    // helper functions

    fn setup_peer_up_ack_event_handler(
        rt: tokio::runtime::Handle,
        connected: Sender<bool>,
    ) -> TransportEventHandler {
        let (event_handler, mut handle) = create_mock_event_handler();

        rt.spawn(async move {
            let (event, rsp) = handle.next_request().await.unwrap();
            if let TransportEvent::PeerUp(_) = event {
                connected.try_send(true).unwrap()
            }
            rsp.send_response(());
        });
        event_handler
    }

    fn setup_message_ack_event_handler(
        rt: tokio::runtime::Handle,
        connected: Sender<TransportPayload>,
    ) -> TransportEventHandler {
        let (event_handler, mut handle) = create_mock_event_handler();

        rt.spawn(async move {
            loop {
                let (event, rsp) = handle.next_request().await.unwrap();
                match event {
                    TransportEvent::Message(msg) => {
                        connected.send(msg.payload).await.expect("Channel busy");
                    }
                    TransportEvent::PeerUp(_) => {}
                    TransportEvent::PeerDown(_) => {}
                };
                rsp.send_response(());
            }
        });
        event_handler
    }

    struct RegistryAndDataProvider {
        data_provider: Arc<ProtoRegistryDataProvider>,
        registry: Arc<FakeRegistryClient>,
    }

    impl RegistryAndDataProvider {
        fn new() -> Self {
            let data_provider = Arc::new(ProtoRegistryDataProvider::new());
            let registry = Arc::new(FakeRegistryClient::new(Arc::clone(&data_provider) as Arc<_>));
            Self {
                data_provider,
                registry,
            }
        }
    }

    fn temp_crypto_component_with_tls_keys_in_registry(
        registry_and_data: &RegistryAndDataProvider,
        node_id: NodeId,
    ) -> TempCryptoComponent {
        let (temp_crypto, tls_pubkey_cert) = TempCryptoComponent::new_with_tls_key_generation(
            Arc::clone(&registry_and_data.registry) as Arc<_>,
            node_id,
        );
        registry_and_data
            .data_provider
            .add(
                &make_crypto_tls_cert_key(node_id),
                REG_V1,
                Some(tls_pubkey_cert.to_proto()),
            )
            .expect("failed to add TLS cert to registry");
        temp_crypto
    }

    // Get a free port on this host to which we can connect transport to.
    fn get_free_localhost_port() -> std::io::Result<u16> {
        let socket = TcpSocket::new_v4()?;
        // This allows transport to bind to this address,
        //  even though the socket is already bound.
        socket.set_reuseport(true)?;
        socket.set_reuseaddr(true)?;
        socket.bind("127.0.0.1:0".parse().unwrap())?;
        Ok(socket.local_addr()?.port())
    }

    // TODO(NET-1182): this test hangs on CI sometimes
    #[test]
    fn test_single_transient_failure_of_tls_client_handshake() {
        with_test_replica_logger(|log| {
            let mut registry_and_data = RegistryAndDataProvider::new();
            let rt = tokio::runtime::Runtime::new().unwrap();
            let rt_handle = rt.handle().clone();

            let crypto_factory_with_single_tls_handshake_client_failures =
                |registry_and_data: &mut RegistryAndDataProvider, node_id: NodeId| {
                    let mut mock_client_tls_handshake = MockTlsHandshake::new();
                    let rt_handle = rt_handle.clone();

                    let crypto = Arc::new(temp_crypto_component_with_tls_keys_in_registry(
                        registry_and_data,
                        node_id,
                    ));

                    mock_client_tls_handshake
                        .expect_perform_tls_client_handshake()
                        .times(1)
                        .returning({
                            move |_tcp_stream: TcpStream,
                                  _server: NodeId,
                                  _registry_version: RegistryVersion| {
                                Err(TlsClientHandshakeError::HandshakeError {
                                    internal_error: "transient".to_string(),
                                })
                            }
                        });

                    mock_client_tls_handshake
                        .expect_perform_tls_client_handshake()
                        .times(1)
                        .returning(
                            move |tcp_stream: TcpStream,
                                  server: NodeId,
                                  registry_version: RegistryVersion| {
                                let rt_handle = rt_handle.clone();
                                let crypto = crypto.clone();

                                tokio::task::block_in_place(move || {
                                    let rt_handle = rt_handle.clone();

                                    rt_handle.block_on(async move {
                                        crypto
                                            .perform_tls_client_handshake(
                                                tcp_stream,
                                                server,
                                                registry_version,
                                            )
                                            .await
                                    })
                                })
                            },
                        );

                    Arc::new(mock_client_tls_handshake) as Arc<dyn TlsHandshake + Send + Sync>
                };

            let crypto_factory = |registry_and_data: &mut RegistryAndDataProvider,
                                  node_id: NodeId| {
                Arc::new(temp_crypto_component_with_tls_keys_in_registry(
                    registry_and_data,
                    node_id,
                )) as Arc<dyn TlsHandshake + Send + Sync>
            };

            let peer1_port = get_free_localhost_port().expect("Failed to get free localhost port");
            let (peer_1, mut mock_handle_peer_1, peer_1_addr) = setup_test_peer(
                log.clone(),
                rt.handle().clone(),
                NODE_1,
                peer1_port,
                REG_V1,
                &mut registry_and_data,
                crypto_factory_with_single_tls_handshake_client_failures,
            );
            let peer2_port = get_free_localhost_port().expect("Failed to get free localhost port");
            let (peer_2, mut mock_handle_peer_2, peer_2_addr) = setup_test_peer(
                log,
                rt.handle().clone(),
                NODE_2,
                peer2_port,
                REG_V1,
                &mut registry_and_data,
                crypto_factory,
            );
            registry_and_data.registry.update_to_latest_version();

            let (connected_1, mut done_1) = channel(1);
            let (connected_2, mut done_2) = channel(1);
            rt.spawn(async move {
                let (event, rsp) = mock_handle_peer_1.next_request().await.unwrap();
                if let TransportEvent::PeerUp(_) = event {
                    connected_1.try_send(true).unwrap()
                }
                rsp.send_response(());
            });
            rt.spawn(async move {
                let (event, rsp) = mock_handle_peer_2.next_request().await.unwrap();
                if let TransportEvent::PeerUp(_) = event {
                    connected_2.try_send(true).unwrap()
                }
                rsp.send_response(());
            });
            assert!(peer_1
                .start_connection(&NODE_ID_2, peer_2_addr, REG_V1)
                .is_ok());

            assert!(peer_2
                .start_connection(&NODE_ID_1, peer_1_addr, REG_V1)
                .is_ok());
            assert_eq!(done_1.blocking_recv(), Some(true));
            assert_eq!(done_2.blocking_recv(), Some(true));
        });
    }

    fn create_mock_event_handler() -> (TransportEventHandler, Handle<TransportEvent, ()>) {
        let (service, handle) = tower_test::mock::pair::<TransportEvent, ()>();

        let infallible_service = tower::service_fn(move |request: TransportEvent| {
            let mut service_clone = service.clone();
            async move {
                service_clone
                    .ready()
                    .await
                    .expect("Mocking Infallible service. Waiting for readiness failed.")
                    .call(request)
                    .await
                    .expect("Mocking Infallible service and can therefore not return an error.");
                Ok::<(), Infallible>(())
            }
        });
        (BoxCloneService::new(infallible_service), handle)
    }

    fn start_connection_between_two_peers(
        rt_handle: tokio::runtime::Handle,
        logger: ReplicaLogger,
        registry_version: RegistryVersion,
        send_queue_size: usize,
        event_handler_1: TransportEventHandler,
        event_handler_2: TransportEventHandler,
    ) -> (Arc<dyn Transport>, Arc<dyn Transport>) {
        // Setup registry and crypto component
        let registry_and_data = RegistryAndDataProvider::new();
        let crypto_1 =
            temp_crypto_component_with_tls_keys_in_registry(&registry_and_data, NODE_ID_1);
        let crypto_2 =
            temp_crypto_component_with_tls_keys_in_registry(&registry_and_data, NODE_ID_2);
        registry_and_data.registry.update_to_latest_version();

        let peer1_port = get_free_localhost_port().expect("Failed to get free localhost port");
        let peer_a_config = TransportConfig {
            node_ip: "127.0.0.1".to_string(),
            listening_port: peer1_port,
            legacy_flow_tag: FLOW_TAG,
            send_queue_size,
        };

        let peer_a = create_transport(
            NODE_ID_1,
            peer_a_config,
            registry_version,
            MetricsRegistry::new(),
            Arc::new(crypto_1),
            rt_handle.clone(),
            logger.clone(),
        );
        peer_a.set_event_handler(event_handler_1);

        let peer2_port = get_free_localhost_port().expect("Failed to get free localhost port");
        let peer_b_config = TransportConfig {
            node_ip: "127.0.0.1".to_string(),
            listening_port: peer2_port,
            legacy_flow_tag: FLOW_TAG,
            send_queue_size,
        };

        let peer_b = create_transport(
            NODE_ID_2,
            peer_b_config,
            registry_version,
            MetricsRegistry::new(),
            Arc::new(crypto_2),
            rt_handle,
            logger,
        );
        peer_b.set_event_handler(event_handler_2);
        let peer_2_addr = SocketAddr::from_str(&format!("127.0.0.1:{}", peer2_port)).unwrap();

        peer_a
            .start_connection(&NODE_ID_2, peer_2_addr, REG_V1)
            .expect("start_connection");

        let peer_1_addr = SocketAddr::from_str(&format!("127.0.0.1:{}", peer1_port)).unwrap();
        peer_b
            .start_connection(&NODE_ID_1, peer_1_addr, REG_V1)
            .expect("start_connection");

        (peer_a, peer_b)
    }
}
