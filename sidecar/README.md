# sidecar — Shard's libp2p transport daemon

Aligned with the repository on 2026-10-08. The Go `go-libp2p` daemon runs beside the Python engine, owns its Ed25519 PeerId identity and moves activation frames over authenticated, encrypted peer streams. It handles peers and bytes, not accounts, payments or model math. See the current [integration boundary](../docs/INTEGRATION.md).

## Build

`go.mod` currently declares Go **1.25.7** and `go-libp2p v0.48.0`; use that file and `go.sum` as the dependency source of truth. `GOTOOLCHAIN=auto` may obtain a required toolchain when the installed Go is older. Pin the actual build/toolchain and retain the binary digest for deployment; the minimum Go directive alone is not a binary provenance proof.

POSIX-shell example for a static Linux/amd64 binary:

```sh
cd sidecar
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOTOOLCHAIN=auto go build -trimpath -o sidecar .
```

A node running that binary does not need Go installed. Match the target OS/architecture; this is not a universal binary for all hosts. Configure an appropriate Go module proxy in your own environment if default resolution is unavailable; old mirror reachability notes are not current availability guarantees.

## Engine tunnels and ring authorization

The interface is **localhost TCP↔libp2p**, not Unix socket/gRPC. Pin the public-mapped libp2p listen port and keep Python engine/tunnel endpoints loopback where that is the intended isolation boundary.

```sh
./sidecar -key /path/to/node.key -listen /ip4/0.0.0.0/tcp/29600 \
  -inbound 127.0.0.1:29610 \
  -forward 127.0.0.1:29611=/ip4/PEER_IP/tcp/PEER_PORT/p2p/PEER_ID \
  -allow PREDECESSOR_PEER_ID
```

Replace the uppercase address/identity placeholders with actual deployment values. `-forward` is repeatable and accepts `localAddr=peerMultiaddr[,peerMultiaddr...]`: supply direct and relay-circuit candidate addresses when needed. Releases predating comma-list support cannot be substituted into such a deployment.

| Flag | Actual behavior |
|---|---|
| `-key PATH` | Persist/reuse the node key and PeerId; keep the file private and out of source control |
| `-allow PEERID` (repeatable) | Only listed authenticated peers may open inbound activation streams; no flags retains open legacy mode |
| `-frame-timeout N` | Per-frame absolute completion deadline from its first prefix byte, default 60 s; pre-frame idle is unbounded; `0` enables legacy raw piping |
| `-announce MULTIADDR` | Advertise a configured public address before automatically detected addresses |
| `-quic` | Add a corresponding QUIC/UDP listener; requires actual UDP reachability |
| `-relay` | Run a circuit-relay-v2 service on a suitable public host |
| `-relays ADDR,ADDR` | Use configured relay peers; reservation/advertised circuit routes support NAT'd nodes |
| `-dht-bootstrap ADDR` (repeatable) | Configure the shard content-routing DHT bootstrap peers |
| `-seed manifest.json=modelDir` | Announce and serve manifest shards via the DHT/block-fetch path |
| `-fetch-cid CID`, `-fetch-out PATH` | One-shot content fetch; `-fetch-size` and `-fetch-timeout` bound its expected bytes and deadline |
| `-prove CHALLENGE`, `-verify peerid,nonce,b64sig` | Identity ownership proof helpers, not compute/hardware attestations |

NAT port mapping, hole punching and relay mechanisms are implemented. They do not guarantee a direct path through every CGNAT or arbitrary firewall. Inspect actual dialable addresses, relay/direct connection logs and route measurements. The application must also authorize the expected coordinator/predecessor/return roles; public contributor registration is compatible with restricted membership of an assigned ring.

## Encryption and message contracts

libp2p provides peer authentication and link encryption (Noise/TLS transport configuration). Python uses `shard.transport`'s length-prefixed JSON headers and raw tensor blobs. The raw TCP `phase0/wire.py` PSK mode remains an explicit alternative; a libp2p ring does not need one shared `SHARD_PSK`.

This transport authenticates bytes and the remote peer, not model correctness. Local plaintext socket access is part of the node's local trust boundary. The new strict GPT-OSS/V4 pipeline sessions additionally bind plan, signer, nonce, role, range and owner fence. M2.5 retains its own compatibility greetings; swapping an import alone does not upgrade every engine to that strict session protocol.

## Connectivity self-test

```sh
# Terminal A: persists one identity and writes a dialable address.

./sidecar -key /tmp/a.key -addrfile /tmp/addrA

# Terminal B: separate identity, one 2 MiB framed round trip.

./sidecar -key /tmp/b.key -peer "$(cat /tmp/addrA)" -size 2097152
```

The output `ROUND-TRIP OK` proves connectivity and frame round-trip for that test. It does not establish WAN geography, NAT success on another host, GPU execution or throughput.

The [2026-06-19 GPT-OSS libp2p record](../docs/receipts/gpt-oss-120b-libp2p-20260619.json) retains the historical raw-TCP/libp2p numerical comparison. It is not a new hardware result for the current binary or strict service. Publisher manifest verification and full model/receipt controls belong to their separate [evidence contracts](../docs/PROOF.md).
