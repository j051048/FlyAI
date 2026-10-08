import socket
import struct
import threading

import pytest
from phase0.v4_link_probe import make_server, measure_route, echo_connection, MAX_PAYLOAD


def test_real_auxiliary_route_retains_raw_samples_and_exact_bytes():
    server = make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = measure_route(f"127.0.0.1:{server.server_address[1]}", samples=5, warmup=1)
        assert len(result["samples"]) == 5
        assert all(row["sent_bytes"] == row["received_bytes"] == 32768 + 36 for row in result["samples"])
        assert 0 < result["p50_rtt_s"] <= result["p95_rtt_s"]
        assert result["counts_inference_acceptance"] is False
        assert "no model/GPU" in result["scope"]
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_wrong_echo_is_rejected_without_a_benchmark_report():
    listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen()
    def fake():
        peer, _ = listener.accept()
        with peer:
            # A complete but wrong length is rejected before any response allocation.
            peer.recv(65536)
            peer.sendall(struct.pack("!I", MAX_PAYLOAD + 1))
    thread = threading.Thread(target=fake, daemon=True); thread.start()
    try:
        with pytest.raises(ValueError, match="different frame size"):
            measure_route(f"127.0.0.1:{listener.getsockname()[1]}", samples=1, warmup=0)
    finally:
        listener.close(); thread.join(timeout=2)


def test_probe_listener_cannot_be_exposed_as_an_unprotected_public_service():
    with pytest.raises(ValueError, match="loopback"):
        make_server("0.0.0.0")
    with pytest.raises(ValueError, match="separate auxiliary port"):
        measure_route("127.0.0.1:29610")


def test_server_rejects_oversized_frame_before_reading_payload():
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!I", MAX_PAYLOAD + 33))
        with pytest.raises(ValueError, match="bounded"):
            echo_connection(right)
    finally:
        left.close(); right.close()
