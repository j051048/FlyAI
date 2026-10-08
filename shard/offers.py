"""Open, signed GPU offers using the existing Ed25519 libp2p identity.

Signatures establish provenance, not hardware attestation. Registration accepts
any identity; execution additionally requires fresh measurements and an exact
model compatibility cohort. No node weights or user prompts enter this registry.
"""
from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

SCHEMA = "shard-node-offer/1"
_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_MODEL_FIELDS = ("model_id", "manifest_sha256", "checkpoint_id", "config_sha256",
                 "quantization", "runtime_abi", "wire_version", "numeric_contract", "n_layers")


class OfferError(ValueError):
    pass


def canonical(body):
    try:
        return json.dumps(body, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise OfferError("finite canonical JSON required") from exc


def _text(value, name, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise OfferError(f"invalid {name}")
    return value


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise OfferError(f"{name} must be a lowercase SHA256")
    return value


def _count(value, name, *, nullable=False):
    if nullable and value is None:
        return value
    if type(value) is not int or value < 0:
        raise OfferError(f"{name} must be nonnegative integer bytes")
    return value


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise OfferError(f"invalid {name}")
    return float(value)


def _b58encode(raw):
    value = int.from_bytes(raw, "big")
    out = ""
    while value:
        value, digit = divmod(value, 58)
        out = _ALPHABET[digit] + out
    return "1" * (len(raw) - len(raw.lstrip(b"\0"))) + out


def peer_id_from_public_key(raw):
    """Ed25519 identity multihash, identical to Go libp2p's inline PeerId."""
    if not isinstance(raw, bytes) or len(raw) != 32:
        raise OfferError("Ed25519 public key must contain 32 bytes")
    return _b58encode(b"\x00\x24\x08\x01\x12\x20" + raw)


def load_sidecar_key(path):
    """Read the existing Go sidecar protobuf key; never create another identity."""
    p = Path(path)
    if p.is_symlink():
        raise OfferError("sidecar key must not be a symlink")
    raw = p.read_bytes()
    if len(raw) != 68 or raw[:4] != b"\x08\x01\x12\x40":
        raise OfferError("expected an existing libp2p Ed25519 private key")
    key = Ed25519PrivateKey.from_private_bytes(raw[4:36])
    if key.public_key().public_bytes_raw() != raw[36:]:
        raise OfferError("sidecar key public half does not match its seed")
    return key


@dataclass(frozen=True)
class ModelCohort:
    model_id: str
    manifest_sha256: str
    checkpoint_id: str
    config_sha256: str
    quantization: str
    runtime_abi: str
    wire_version: str
    numeric_contract: str
    n_layers: int

    def __post_init__(self):
        for name in _MODEL_FIELDS[:-1]:
            (_hash if name.endswith("sha256") else _text)(getattr(self, name), name)
        if type(self.n_layers) is not int or not 1 <= self.n_layers <= 100000:
            raise OfferError("invalid model layer count")

    @property
    def cohort_id(self):
        return hashlib.sha256(b"shard-model-cohort/1\0" + canonical(asdict(self))).hexdigest()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(_MODEL_FIELDS):
            raise OfferError("exact model cohort descriptor required")
        return cls(**value)


def model_cohort_id(value):
    return (value if isinstance(value, ModelCohort) else ModelCohort.from_dict(value)).cohort_id


def signing_message(body):
    unsigned = {k: v for k, v in body.items() if k != "signature"}
    return "shard-node-offer/1:" + hashlib.sha256(canonical(unsigned)).hexdigest()


def sign_offer(body, key):
    result = copy.deepcopy(body)
    pub = key.public_key().public_bytes_raw()
    result.update(schema=SCHEMA, peer_id=peer_id_from_public_key(pub),
                  public_key=base64.b64encode(pub).decode())
    result["node_id"] = f"{result['peer_id']}/{result['gpu_uuid']}"
    result["signature"] = base64.b64encode(key.sign(signing_message(result).encode())).decode()
    return result


def _offer_trace(offer, cohort_id, profile, *, now):
    """Check a trace in its recorded shape before it can enter a planner pool.

    Missing identity bindings allow registration, but never eligibility. The
    caller cannot substitute a scalar speed for an unbound chunk observation.
    """
    from .locality import timestamp, number
    from .planning_cost import stage_observation
    trace = profile["stage_trace"]
    if not isinstance(trace, dict):
        raise ValueError("stage trace must be an object")
    frame_tokens = trace.get("frame_tokens", 1)
    if type(frame_tokens) is not int or not 1 <= frame_tokens <= 256:
        raise ValueError("invalid trace frame_tokens")
    workload = {"frame_tokens": frame_tokens}
    if frame_tokens > 1:
        context = trace.get("context_tokens")
        if type(context) is not int or not 1 <= context <= 1_000_000:
            raise ValueError("invalid trace context_tokens")
        if trace.get("warmness") not in ("cold", "warm"):
            raise ValueError("invalid trace warmness")
        workload.update(context_tokens=context, warmness=trace["warmness"])
    age = now - timestamp(trace["measured_at"])
    ttl = number(trace["ttl_s"], "trace ttl_s", minimum=1e-9)
    observed = stage_observation({"id": offer["node_id"], "gpu_uuid": offer["gpu_uuid"],
        "cohort_id": cohort_id, "runtime_config_sha256": profile.get("runtime_config_sha256"),
        "stage_trace": trace}, trace["layer_end"] - trace["layer_start"], profile.get("layer_ms") or 1,
        now=now, start=trace["layer_start"], workload=workload, require_fresh_chunk=False)
    return observed, -30 <= age <= ttl


def validate_offer(body, *, now=None, max_ttl_s=300, max_skew_s=30):
    now = time.time() if now is None else now
    if not isinstance(body, dict) or body.get("schema") != SCHEMA or len(canonical(body)) > 256 * 1024:
        raise OfferError("unsupported or oversized offer")
    required = {"schema", "peer_id", "public_key", "node_id", "gpu_uuid", "memory_domain_id",
                "endpoints", "resources", "models", "sequence", "issued_at", "ttl_s", "signature"}
    allowed = required | {"region", "zone", "public_ip", "host_id", "lease_endpoint", "gpu_model"}
    if not required <= set(body) or set(body) - allowed:
        raise OfferError("missing or unknown offer fields")
    _text(body["gpu_uuid"], "gpu_uuid")
    _text(body["memory_domain_id"], "memory_domain_id")
    for field in ("region", "zone", "public_ip", "host_id", "gpu_model", "lease_endpoint"):
        if body.get(field) is not None:
            _text(body[field], field, 2048)
    if type(body["sequence"]) is not int or body["sequence"] < 1:
        raise OfferError("positive sequence required")
    stamp, ttl = _number(body["issued_at"], "issued_at"), _number(body["ttl_s"], "ttl_s", positive=True)
    if ttl > max_ttl_s or stamp > now + max_skew_s or stamp + ttl <= now:
        raise OfferError("offer is expired or outside the allowed time window")
    eps = body["endpoints"]
    if not isinstance(eps, list) or not 1 <= len(eps) <= 16:
        raise OfferError("reachable endpoint candidates required")
    for endpoint in eps:
        _text(endpoint, "endpoint", 2048)
    resources = body["resources"]
    names = {"available_vram_bytes", "available_ram_bytes", "pinnable_ram_bytes", "available_disk_bytes", "measured_at"}
    if not isinstance(resources, dict) or set(resources) != names:
        raise OfferError("exact resource measurement fields required")
    for name in names - {"measured_at"}:
        _count(resources[name], name, nullable=True)
    measured = _number(resources["measured_at"], "resource measured_at")
    if measured > now + max_skew_s or measured + max_ttl_s <= now:
        raise OfferError("resource measurement is stale")
    ram, pin = resources["available_ram_bytes"], resources["pinnable_ram_bytes"]
    if ram is not None and pin is not None and pin > ram:
        raise OfferError("pinned capacity is a subset of host RAM")
    models = body["models"]
    if not isinstance(models, list) or len(models) > 64:
        raise OfferError("models must be a bounded list")
    cohorts = set()
    for entry in models:
        if (not isinstance(entry, dict) or not {"cohort", "profile", "measured_at"} <= set(entry)
                or set(entry) - {"cohort", "profile", "measured_at", "calibrations"}):
            raise OfferError("invalid model capability")
        cohort = ModelCohort.from_dict(entry["cohort"])
        if cohort.cohort_id in cohorts:
            raise OfferError("duplicate model capability")
        cohorts.add(cohort.cohort_id)
        cap_stamp = _number(entry["measured_at"], "capability measured_at")
        if cap_stamp > now + max_skew_s:
            raise OfferError("model measurement is from the future")
        profile = entry["profile"]
        permitted = {"layer_ms", "cap_layers", "layer_vram_mb", "total_vram_mb", "h2d_gbps", "up_mbps",
                     "dma_exposed_ms_per_layer", "expert_misses_per_layer", "dma_overlap_fraction", "stage_trace",
                     "runtime_config_sha256"}
        if not isinstance(profile, dict) or set(profile) - permitted:
            raise OfferError("invalid measured profile fields")
        for key, value in profile.items():
            if key == "runtime_config_sha256":
                _hash(value, key)
            elif key == "stage_trace":
                if not isinstance(value, dict):
                    raise OfferError("stage_trace must be an object")
            elif key == "cap_layers":
                _count(value, key)
                if value > cohort.n_layers:
                    raise OfferError("layer capacity exceeds model")
            elif value is not None:
                _number(value, key, positive=key in {"layer_ms", "layer_vram_mb", "h2d_gbps", "up_mbps"})
        overlap = profile.get("dma_overlap_fraction")
        if overlap is not None and overlap > 1:
            raise OfferError("DMA overlap must be within [0,1]")
        if profile.get("stage_trace") is not None:
            # Validate at admission so one signed malformed observation cannot
            # make planning fail for every honest node in the same cohort.
            try:
                trace = profile["stage_trace"]
                _, fresh = _offer_trace(body, cohort.cohort_id, profile, now=now)
                if not fresh:
                    raise ValueError("stage trace expired or is from the future")
                if trace["layer_end"] > cohort.n_layers:
                    raise ValueError("trace lies outside the model")
            except (ValueError, TypeError, KeyError) as exc:
                raise OfferError("invalid or mismatched stage trace") from exc
        calibrations = entry.get("calibrations", [])
        if not isinstance(calibrations, list) or len(calibrations) > 64:
            raise OfferError("bounded stage calibrations required")
        seen_ranges = set()
        for record in calibrations:
            try:
                from .resources import PlacementRequirements
                if not isinstance(record, dict) or set(record) != {"requirements", "runtime_config"}:
                    raise ValueError("invalid calibration record")
                req = PlacementRequirements.from_dict(record["requirements"])
                cfg = record["runtime_config"]
                if (req.model_id != cohort.model_id or req.provenance.checkpoint_id != cohort.checkpoint_id
                        or req.provenance.node_id != body["node_id"] or req.layer_end > cohort.n_layers
                        or not isinstance(cfg, dict) or type(cfg.get("lo")) is not int or type(cfg.get("hi")) is not int
                        or cfg.get("lo") != req.layer_start or cfg.get("hi") != req.layer_end
                        or type(cfg.get("head")) is not bool or type(cfg.get("tail")) is not bool
                        or cfg["head"] != (req.layer_start == 0) or cfg["tail"] != (req.layer_end == cohort.n_layers)
                        or hashlib.sha256(canonical(cfg)).hexdigest() != req.provenance.runtime_config_sha256):
                    raise ValueError("calibration identity or runtime differs from offer")
                if not isinstance(cfg.get("environment", {}), dict) or any(not isinstance(k, str) for k in cfg.get("environment", {})):
                    raise ValueError("runtime environment must be a string-keyed mapping")
                if any(k.startswith("SHARD_") for k in cfg.get("environment", {})):
                    raise ValueError("private provisioning cannot appear in a public calibration")
                bounds = req.layer_start, req.layer_end
                if type(cfg.get("n_layers", cohort.n_layers)) is not int or cfg.get("n_layers", cohort.n_layers) != cohort.n_layers:
                    raise ValueError("calibration model depth differs from cohort")
                for key in ("stage", "index", "nstages"):
                    if key in cfg and (type(cfg[key]) is not int or cfg[key] < (1 if key == "nstages" else 0)):
                        raise ValueError("calibration stage geometry must be integer")
                index = cfg.get("stage", cfg.get("index"))
                if "stage" in cfg and "index" in cfg and cfg["stage"] != cfg["index"]:
                    raise ValueError("calibration stage/index differs")
                if index is not None and "nstages" in cfg and (index >= cfg["nstages"] or
                        cfg["head"] != (index == 0) or cfg["tail"] != (index == cfg["nstages"] - 1)):
                    raise ValueError("calibration roles differ from stage geometry")
                identity = (*bounds, req.provenance.runtime_config_sha256)
                if identity in seen_ranges:
                    raise ValueError("duplicate stage calibration")
                seen_ranges.add(identity)
            except (ValueError, TypeError, KeyError) as exc:
                raise OfferError("invalid stage calibration") from exc
    try:
        pub = base64.b64decode(body["public_key"], validate=True)
        peer_id = peer_id_from_public_key(pub)
        if body["peer_id"] != peer_id or body["node_id"] != f"{peer_id}/{body['gpu_uuid']}":
            raise OfferError("node identity is not bound to its libp2p key and GPU")
        Ed25519PublicKey.from_public_bytes(pub).verify(
            base64.b64decode(body["signature"], validate=True), signing_message(body).encode())
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise OfferError("offer signature or identity failed verification") from exc
    return copy.deepcopy(body)


class OfferRegistry:
    """Durable bounded reference registry; replay tombstones survive offer expiry."""
    def __init__(self, path=":memory:", *, clock=time.time, max_nodes=10000, max_ttl_s=300,
                 sequence_retention_s=86400):
        if type(max_nodes) is not int or max_nodes < 1:
            raise ValueError("positive registry capacity required")
        self.clock, self.max_nodes, self.max_ttl_s = clock, max_nodes, max_ttl_s
        self.sequence_retention_s = _number(sequence_retention_s, "sequence retention", positive=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=10)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS offers(node_id TEXT PRIMARY KEY, sequence INTEGER NOT NULL, digest TEXT NOT NULL, expires REAL NOT NULL, body TEXT NOT NULL)")

    def announce(self, body):
        offer = validate_offer(body, now=self.clock(), max_ttl_s=self.max_ttl_s)
        raw = canonical(offer)
        digest = hashlib.sha256(raw).hexdigest()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                now = self.clock()
                self._db.execute("UPDATE offers SET body='' WHERE expires<=? AND body!=''", (now,))
                self._db.execute("DELETE FROM offers WHERE expires<=?", (now - self.sequence_retention_s,))
                old = self._db.execute("SELECT sequence,digest,expires FROM offers WHERE node_id=?", (offer["node_id"],)).fetchone()
                if old:
                    if offer["sequence"] < old[0] or (offer["sequence"] == old[0] and digest != old[1]):
                        raise OfferError("replayed or conflicting offer sequence")
                    if offer["sequence"] == old[0]:
                        self._db.execute("COMMIT")
                        return {"node_id": offer["node_id"], "unchanged": True}
                if (old is None or old[2] <= now) and self._db.execute("SELECT count(*) FROM offers WHERE expires>?", (now,)).fetchone()[0] >= self.max_nodes:
                    raise OfferError("registry capacity reached")
                # Bound replay metadata independently of live registrations.
                # Evicted packets are already expired and cannot be replayed.
                excess = self._db.execute("SELECT count(*) FROM offers").fetchone()[0] - self.max_nodes * 4 + 1
                if excess > 0:
                    self._db.execute("DELETE FROM offers WHERE node_id IN (SELECT node_id FROM offers WHERE expires<=? AND node_id!=? ORDER BY expires LIMIT ?)",
                                     (now, offer["node_id"], excess))
                self._db.execute("INSERT INTO offers VALUES(?,?,?,?,?) ON CONFLICT(node_id) DO UPDATE SET sequence=excluded.sequence,digest=excluded.digest,expires=excluded.expires,body=excluded.body",
                                 (offer["node_id"], offer["sequence"], digest, offer["issued_at"] + offer["ttl_s"], raw.decode()))
                self._db.execute("COMMIT")
            except BaseException:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise
        return {"node_id": offer["node_id"], "unchanged": False}

    def active(self):
        with self._lock:
            rows = self._db.execute("SELECT body FROM offers WHERE expires>? ORDER BY node_id", (self.clock(),)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def snapshot(self, cohort_id):
        _hash(cohort_id, "cohort_id")
        now, nodes = self.clock(), []
        for offer in self.active():
            resources = offer["resources"]
            if resources["measured_at"] + self.max_ttl_s <= now or resources["available_vram_bytes"] is None:
                continue
            for model in offer["models"]:
                if model_cohort_id(model["cohort"]) != cohort_id or model["measured_at"] + self.max_ttl_s <= now:
                    continue
                profile = copy.deepcopy(model["profile"])
                # A fresh signed announcement is not a calibration of an absent speed.
                if profile.get("layer_ms") is None and profile.get("stage_trace") is None:
                    continue
                if profile.get("stage_trace") is not None:
                    try:
                        observed, fresh = _offer_trace(offer, cohort_id, profile, now=now)
                    except (ValueError, TypeError, KeyError):
                        # An old persisted record cannot poison all honest offers.
                        continue
                    if not fresh or observed["source"] != "fresh_stage_trace":
                        continue
                def mib(name):
                    value = resources[name]
                    return None if value is None else value / (1024 * 1024)
                templates = {}
                if "calibrations" in model:
                    from .resources import PlacementRequirements, NodeResources, evaluate_fit
                    from .locality import timestamp
                    allowed = []
                    capacity = NodeResources(**{key: resources[key] for key in (
                        "available_vram_bytes", "available_ram_bytes", "pinnable_ram_bytes", "available_disk_bytes")})
                    for record in model["calibrations"]:
                        req = PlacementRequirements.from_dict(record["requirements"])
                        age = now - timestamp(req.provenance.measured_at)
                        if not -30 <= age <= self.max_ttl_s or not evaluate_fit(req, capacity)["fits"]:
                            continue
                        config = record["runtime_config"]
                        span = {"lo": req.layer_start, "hi": req.layer_end,
                            "head": config["head"], "tail": config["tail"],
                            "runtime_config_sha256": req.provenance.runtime_config_sha256,
                            "gpu_bytes": req.gpu.peak_bytes, "host_bytes": req.host.peak_bytes,
                            "pinned_bytes": req.host.pinned_bytes}
                        if "stage" in config or "index" in config:
                            span["stage_index"] = config.get("stage", config.get("index"))
                        if "nstages" in config:
                            span["nstages"] = config["nstages"]
                        allowed.append(span)
                    templates = {"allowed_spans": allowed, "resource_capacity": {
                        key: resources[key] for key in ("available_vram_bytes", "available_ram_bytes", "pinnable_ram_bytes")}}
                nodes.append({**profile, "id": offer["node_id"], "cohort_id": cohort_id, "peer_id": offer["peer_id"],
                              "gpu_uuid": offer["gpu_uuid"], "memory_domain_id": offer["memory_domain_id"],
                              "host_id": offer.get("host_id"), "public_ip": offer.get("public_ip"),
                              "region": offer.get("region"), "zone": offer.get("zone"),
                              "free_vram_mb": mib("available_vram_bytes"), "free_ram_mb": mib("available_ram_bytes"),
                              "pinnable_ram_mb": mib("pinnable_ram_bytes"),
                              "offer_sequence": offer["sequence"], "offer_expires_at": offer["issued_at"] + offer["ttl_s"],
                              "measurement_source": "signed_operator_report", "hardware_attested": False, **templates})
        return nodes

    def get(self, node_id):
        with self._lock:
            row = self._db.execute("SELECT body FROM offers WHERE node_id=? AND expires>?", (node_id, self.clock())).fetchone()
        if row is None:
            raise OfferError("node offer is unavailable")
        return json.loads(row[0])

    def close(self):
        with self._lock:
            self._db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sign = commands.add_parser("sign")
    sign.add_argument("--offer", required=True)
    sign.add_argument("--sidecar-key", required=True)
    sign.add_argument("--out", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--offer", required=True)
    export = commands.add_parser("export-receipt-key", help="export the same sidecar identity in the Python receipt key format")
    export.add_argument("--sidecar-key", required=True)
    export.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "export-receipt-key":
        from .manifest import save_key
        target = Path(args.out)
        if target.exists() or target.is_symlink():
            parser.error("receipt key output already exists")
        key = load_sidecar_key(args.sidecar_key)
        save_key(key, str(target))
        print(json.dumps({"peer_id": peer_id_from_public_key(key.public_key().public_bytes_raw()), "path": str(target)}))
        return
    body = json.loads(Path(args.offer).read_text(encoding="utf-8-sig"))
    if args.command == "sign":
        body = sign_offer(body, load_sidecar_key(args.sidecar_key))
        validate_offer(body)
        Path(args.out).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    else:
        validate_offer(body)
    print(json.dumps({"node_id": body["node_id"], "signature_verified": True, "hardware_attested": False}))


if __name__ == "__main__":
    main()
