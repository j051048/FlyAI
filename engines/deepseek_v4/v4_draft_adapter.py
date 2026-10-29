"""Offline teacher-output distillation of a draft-only low-rank logit residual.

No native packed parameter is trained or replaced, including the shared main
LM head. Exported adapters are local operator artifacts, not inference proofs.
The verifier and its accept/rollback rule remain authoritative.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

SCHEMA = "v4-draft-residual-adapter/1"
BINDING = "v4-draft-training-binding/1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def binding(tail, *, reference=False, fixture_tokenizer=None):
    if hasattr(tail, "_draft_base_binding"):
        return deepcopy(tail._draft_base_binding)
    proof = getattr(tail.stage, "_artifact_verification", None)
    if proof is None:
        if not reference or str(tail.device) != "cpu" or not isinstance(fixture_tokenizer, str) or len(fixture_tokenizer) != 64:
            raise ValueError("draft adapters require verified native artifacts, or explicit CPU fixture identity")
        from v4_numeric_diagnostics import _model_identity, tensor_identity
        model = _model_identity(tail.stage)["model"]
        checkpoint = digest([model["fixture_weights_sha256"],
            [[name,tensor_identity(value)] for name,value in tail.mtp.state_dict().items()]])
        tokenizer = fixture_tokenizer
    else:
        from shard.weight_artifacts import verified_stage_artifact_descriptor
        verified_stage_artifact_descriptor(proof)
        checkpoint = proof["checkpoint_id"].removeprefix("tensor-sha256:")
        assets = {name: value["sha256"] for name, value in proof["catalog"]["assets"].items()
                  if name.startswith(("tokenizer", "vocab", "merges"))}
        if not assets:
            raise ValueError("draft training binding needs original tokenizer assets")
        tokenizer = digest(assets)
    return {"schema": BINDING, "checkpoint_sha256": checkpoint,
        "tokenizer_sha256": tokenizer, "args_sha256": digest(vars(tail.args)),
        "vocab_size": tail.args.vocab_size, "hidden_dim": tail.args.dim,
        "reference": reference}


def validate_binding(value):
    keys = {"schema", "checkpoint_sha256", "tokenizer_sha256", "args_sha256", "vocab_size", "hidden_dim", "reference"}
    if not isinstance(value, dict) or set(value) != keys or value["schema"] != BINDING or type(value["reference"]) is not bool:
        raise ValueError("complete draft training binding required")
    for name in ("checkpoint_sha256", "tokenizer_sha256", "args_sha256"):
        if not isinstance(value[name], str) or len(value[name]) != 64 or any(c not in "0123456789abcdef" for c in value[name]):
            raise ValueError("draft identities must be exact SHA256")
    if any(type(value[k]) is not int or not 1 <= value[k] <= limit for k, limit in
           (("vocab_size", 262144), ("hidden_dim", 65536))):
        raise ValueError("bounded native vocabulary and hidden dimension required")
    return value


def coverage(tokens, proposed_ids, *, target_ids=()):
    """Decoded token-script audit plus exact observed target-ID coverage."""
    if (not isinstance(tokens, dict) or len(tokens) > 262144 or not tokens
            or any(type(k) is not int or k < 0 or not isinstance(v, str) or len(v) > 4096 for k,v in tokens.items())):
        raise ValueError("bounded decoded tokenizer table required")
    proposed = set(proposed_ids)
    if any(type(value) is not int or value not in tokens for value in proposed):
        raise ValueError("draft vocabulary is outside the tokenizer")
    counts = {key: {"total": 0, "covered": 0} for key in ("cjk", "latin", "other")}
    for token, text in tokens.items():
        category = ("cjk" if any('\u3400' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af' for c in text)
                    else "latin" if any('a' <= c.lower() <= 'z' for c in text) else "other")
        counts[category]["total"] += 1
        counts[category]["covered"] += token in proposed
    targets = list(target_ids)
    if any(type(value) is not int or value not in tokens for value in targets):
        raise ValueError("workload target outside tokenizer")
    return {"schema": "v4-draft-vocabulary-audit/1", "scripts": counts,
        "observed_targets": len(targets), "uncovered_targets": sum(x not in proposed for x in targets),
        "decoded_script_scope": "single_token_decode_lower_bound; byte_fragments_may_be_other",
        "gpu_qualified": False}


class Residual(torch.nn.Module):
    def __init__(self, dim, vocab, rank, *, seed=0):
        super().__init__()
        if type(rank) is not int or not 1 <= rank <= min(64, dim):
            raise ValueError("draft adapter rank must be 1..min(64, hidden)")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.down = torch.nn.Parameter(torch.randn(rank, dim, generator=generator, device="cpu") * .01)
        self.up = torch.nn.Parameter(torch.zeros(vocab, rank, device="cpu"))

    def forward(self, value):
        return F.linear(F.linear(value.float(), self.down), self.up)


def collect(tail, records, *, host_bytes):
    """Extract first-draft-slot examples from original native MTP forwards.

    One record is one independently exported conversation: prompt_taps [1,P,H],
    anchor_ids [N], main_taps [1,N,H], teacher_logits [N,V]. Tap at p pairs with
    the committed anchor at p+1; teacher distribution predicts p+2. Only slot 0
    is collected, whose Markov parent is always known, rather than incorrectly
    treating a rejected draft suffix as a teacher-forced history.
    """
    if getattr(tail.mtp[-1], "_draft_residual", None) is not None or tail.pos is not None:
        raise ValueError("collection needs an idle unadapted native drafter")
    if not isinstance(records, list) or not 4 <= len(records) <= 128 or type(host_bytes) is not int or host_bytes <= 0:
        raise ValueError("four or more bounded independent conversation exports required")
    total = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {"prompt_taps","anchor_ids","main_taps","teacher_logits"}:
            raise ValueError("teacher-forced MTP export fields differ")
        if any(not torch.is_tensor(v) for v in record.values()) or record["prompt_taps"].ndim != 3:
            raise ValueError("teacher export tensor layout differs")
        p, n = record["prompt_taps"].shape[1], record["anchor_ids"].numel()
        shapes={"prompt_taps":(1,p,tail.hidden_dim),"anchor_ids":(n,),
                "main_taps":(1,n,tail.hidden_dim),"teacher_logits":(n,tail.args.vocab_size)}
        if not 2 <= n <= 16384 or p < 1 or p+n+tail.block_size+1 > tail.args.max_seq_len:
            raise ValueError("teacher-forced collector leaves insufficient native horizon")
        for name,value in record.items():
            dtype=torch.long if name=="anchor_ids" else torch.float32 if name=="teacher_logits" else tail.dtype
            if not torch.is_tensor(value) or value.device.type!="cpu" or tuple(value.shape)!=shapes[name] or value.dtype!=dtype:
                raise ValueError("teacher export tensor layout differs")
            total+=value.numel()*value.element_size()
    rows=sum(record["anchor_ids"].numel() for record in records)
    required=total+2*rows*(tail.args.dim+2*tail.args.vocab_size)*4+(8 << 20)
    if rows>65536 or required>host_bytes:
        raise ValueError("teacher collection input/output exceeds host quota")
    result={name:[] for name in ("features","base_logits","target_ids","group_ids","teacher_logits")}
    try:
        for group,record in enumerate(records):
            if any(not bool(torch.isfinite(v).all()) for v in record.values()):
                raise ValueError("nonfinite teacher export")
            if bool((record["anchor_ids"]<0).any()) or bool((record["anchor_ids"]>=tail.args.vocab_size).any()):
                raise ValueError("teacher anchor outside tokenizer")
            tail.reset()
            tail.prefill(record["anchor_ids"][:1],record["prompt_taps"])
            start=record["prompt_taps"].shape[1]
            with torch.no_grad():
                for index,token in enumerate(record["anchor_ids"]):
                    ids=token.reshape(1).to(tail.device)
                    mh=record["main_taps"][:,index:index+1].to(tail.device)
                    h,main=tail.mtp[0].forward_embed(mh,ids)
                    for block in tail.mtp:
                        h=block(h,start+index,ids,main)
                    last=tail.mtp[-1]
                    feature=last.norm(last.hc_head(h,last.hc_head_fn,last.hc_head_scale,last.hc_head_base))
                    outputs=last.forward_head(h,ids)
                    result["features"].append(feature[:,0].float().cpu())
                    result["base_logits"].append(outputs[1][:,0].float().cpu())
                    teacher=record["teacher_logits"][index:index+1]
                    result["teacher_logits"].append(teacher)
                    result["target_ids"].append(teacher.argmax(-1))
                    result["group_ids"].append(torch.tensor([group],dtype=torch.long,device="cpu"))
                    tail._pos=start+index
        return {name:torch.cat(values,0).contiguous() for name,values in result.items()}
    finally:
        tail.reset()


def train(data, identity, *, rank=8, steps=100, batch_size=16, lr=.01, seed=0, host_bytes):
    """Hard teacher-target CE plus optional teacher-distribution KL distillation.

    Group IDs define held-out conversations, not random token rows. Native draft
    hidden features and pre-residual (Markov-biased) logits must be exported by
    an independently checked teacher-forced run with this exact binding.
    """
    validate_binding(identity)
    required = {"features", "base_logits", "target_ids", "group_ids"}
    if not isinstance(data, dict) or not required.issubset(data) or set(data) - required - {"teacher_logits"}:
        raise ValueError("draft training needs frozen features/logits/teacher targets/group IDs")
    if not torch.is_tensor(data["features"]) or data["features"].ndim != 2:
        raise ValueError("frozen CPU training tensor layout differs")
    n = data["features"].shape[0]
    shapes = {"features": (n, identity["hidden_dim"]), "base_logits": (n, identity["vocab_size"]),
              "target_ids": (n,), "group_ids": (n,), "teacher_logits": (n, identity["vocab_size"])}
    if (type(rank) is not int or not 1 <= rank <= min(64, identity["hidden_dim"])
            or not 8 <= n <= 65536 or type(steps) is not int or not 1 <= steps <= 10000
            or type(batch_size) is not int or not 1 <= batch_size <= 128
            or type(host_bytes) is not int or host_bytes <= 0
            or type(lr) not in (float,int) or not math.isfinite(lr) or not 0 < lr <= 1):
        raise ValueError("bounded offline training resources required")
    for name, value in data.items():
        if (not torch.is_tensor(value) or value.device.type != "cpu" or tuple(value.shape) != shapes[name]
                or value.dtype != (torch.long if name in ("target_ids", "group_ids") else torch.float32)):
            raise ValueError("frozen CPU training tensor layout differs")
    model_bytes = (identity["hidden_dim"] + identity["vocab_size"]) * rank * 4
    requirement = sum(x.numel()*x.element_size() for x in data.values()) + 5 * model_bytes + batch_size * identity["vocab_size"] * 32 + (8 << 20)
    if requirement > host_bytes:
        raise ValueError("draft training exceeds declared host quota before optimizer allocation")
    if any(not bool(torch.isfinite(x).all()) for x in data.values()):
        raise ValueError("nonfinite teacher training tensors")
    targets = data["target_ids"]
    if bool((targets < 0).any()) or bool((targets >= identity["vocab_size"]).any()):
        raise ValueError("teacher targets outside original vocabulary")
    groups = sorted(set(data["group_ids"].tolist()), key=lambda x: digest([seed, x]))
    if len(groups) < 4:
        raise ValueError("at least four conversation groups required for held-out validation")
    held = set(groups[:max(1, len(groups)//4)])
    validation = torch.tensor([i for i,g in enumerate(data["group_ids"].tolist()) if g in held], dtype=torch.long)
    training = torch.tensor([i for i,g in enumerate(data["group_ids"].tolist()) if g not in held], dtype=torch.long)
    if min(len(training), len(validation)) < 2:
        raise ValueError("independent train/held-out groups need at least two examples")
    model = Residual(identity["hidden_dim"], identity["vocab_size"], rank, seed=seed)

    def objective(rows):
        prediction = data["base_logits"][rows] + model(data["features"][rows])
        loss = F.cross_entropy(prediction, targets[rows])
        if "teacher_logits" in data:
            loss = loss + F.kl_div(F.log_softmax(prediction, -1), F.softmax(data["teacher_logits"][rows], -1), reduction="batchmean")
        return loss

    def evaluate():
        with torch.no_grad():
            total = sum(float(objective(validation[i:i+batch_size])) * len(validation[i:i+batch_size])
                        for i in range(0, len(validation), batch_size))
        return total / len(validation)

    before = evaluate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for step in range(steps):
        rows = training[torch.randperm(len(training), generator=generator)[:batch_size]]
        optimizer.zero_grad(set_to_none=True)
        loss = objective(rows)
        if not bool(torch.isfinite(loss)):
            raise ValueError("draft distillation diverged")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 0 or step == steps-1:
            history.append(float(loss.detach()))
    after = evaluate()
    return model.eval(), {"schema": SCHEMA, "binding": deepcopy(identity), "rank": rank,
        "seed": seed, "steps": steps, "training_rows": len(training), "heldout_rows": len(validation),
        "training_group_sha256": digest([x for x in groups if x not in held]),
        "heldout_group_sha256": digest(sorted(held)), "heldout_loss_before": before,
        "heldout_loss_after": after, "training_loss_samples": history,
        "heldout_no_regression": after <= before, "host_reserved_bytes": requirement,
        "gpu_qualified": False, "release_qualified": False, "teacher_execution_proven": False}


def export(directory, model, report):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / "adapter.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name,value in model.state_dict().items()}, str(weights))
    manifest = {"schema": SCHEMA, "binding": report["binding"], "rank": report["rank"],
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "weights_bytes": weights.stat().st_size,
        "training_report_sha256": digest(report), "heldout_no_regression": report["heldout_no_regression"],
        "release_qualified": False}
    (directory / "training-report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return manifest


def load(directory, expected, *, max_bytes):
    from safetensors.torch import load_file
    validate_binding(expected)
    directory = Path(directory)
    meta = directory / "manifest.json"
    if meta.stat().st_size > 65536:
        raise ValueError("oversized draft adapter manifest")
    manifest = json.loads(meta.read_text(encoding="utf-8"))
    if (set(manifest) != {"schema", "binding", "rank", "weights_sha256", "weights_bytes", "training_report_sha256",
                          "heldout_no_regression", "release_qualified"} or manifest["schema"] != SCHEMA
            or manifest["binding"] != expected or manifest["heldout_no_regression"] is not True
            or manifest["release_qualified"] is not False or type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("draft adapter identity/held-out policy differs")
    rank=manifest["rank"]
    if (type(rank) is not int or not 1 <= rank <= min(64,expected["hidden_dim"])
            or (expected["hidden_dim"]+expected["vocab_size"])*rank*4 > max_bytes):
        raise ValueError("draft adapter shape exceeds declared weight quota")
    path = directory / "adapter.safetensors"
    if (path.stat().st_size > max_bytes or path.stat().st_size != manifest["weights_bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["weights_sha256"]):
        raise ValueError("draft adapter weight bytes changed or exceed quota")
    report = directory / "training-report.json"
    if report.stat().st_size > 65536:
        raise ValueError("draft adapter report changed")
    evidence = json.loads(report.read_text(encoding="utf-8"))
    if (digest(evidence) != manifest["training_report_sha256"] or evidence.get("binding") != expected
            or evidence.get("rank") != manifest["rank"] or evidence.get("heldout_no_regression") is not True
            or evidence.get("release_qualified") is not False
            or any(type(evidence.get(k)) not in (float,int) or not math.isfinite(evidence[k]) for k in
                   ("heldout_loss_before", "heldout_loss_after"))
            or evidence["heldout_loss_after"] > evidence["heldout_loss_before"]):
        raise ValueError("draft adapter report changed")
    values = load_file(str(path),device="cpu")
    shapes={"down":(rank,expected["hidden_dim"]),"up":(expected["vocab_size"],rank)}
    if set(values)!=set(shapes) or any(tuple(v.shape)!=shapes[k] or v.dtype!=torch.float32 for k,v in values.items()):
        raise ValueError("draft adapter tensor layout differs")
    model = Residual(expected["hidden_dim"], expected["vocab_size"], manifest["rank"])
    model.load_state_dict(values, strict=True)
    if any(not bool(torch.isfinite(v).all()) for v in values.values()):
        raise ValueError("draft adapter nonfinite weights")
    return model.eval(), manifest


def _head(self, value, input_ids):
    model = self._draft_residual
    x = self.hc_head(value, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
    normalized = self.norm(x)
    logits = self.head(normalized, full_logits=True)
    residual = model(normalized)
    output = input_ids.new_empty(input_ids.shape[0], self.block_size + 1)
    output[:, 0] = input_ids
    embeddings = []
    from v4_stage import ref
    for index in range(self.block_size):
        bias, embed = self.markov_head(output[:, index])
        logits[:, index].add_(bias)
        logits[:, index].add_(residual[:, index])
        output[:, index+1] = ref().sample(logits[:, index], self.temperature)
        embeddings.append(embed)
    return output, logits, self.confidence_head(x, torch.stack(embeddings, dim=1))


def install(tail, model, identity):
    validate_binding(identity)
    if binding(tail, reference=identity["reference"], fixture_tokenizer=identity["tokenizer_sha256"]) != identity:
        raise ValueError("draft residual is bound to another native model/tokenizer")
    chain = getattr(tail, "_draft_chain", None)
    if (hasattr(tail.mtp[-1], "_draft_residual") or getattr(tail, "_fast_head_graph", None) is not None
            or chain is not None and (chain._stamp is not None or chain.entries)):
        raise ValueError("draft adapter must bind once before graph capture")
    expected = (tail.args.vocab_size, tail.args.dim)
    if model.up.shape[0] != expected[0] or model.down.shape[1] != expected[1]:
        raise ValueError("draft residual dimensions differ")
    tail._draft_base_binding = deepcopy(identity)
    last = tail.mtp[-1]
    last._draft_residual = model.to(device=tail.device)
    last.forward_head = MethodType(_head, last)
    tail._draft_adapter_status = {"enabled": True, "schema": SCHEMA, "binding_sha256": digest(identity),
                                 "rank": model.down.shape[0], "release_qualified": False}


def configured(tail):
    import os
    path = os.environ.get("V4_DRAFT_ADAPTER_DIR", "")
    budget = int(os.environ.get("V4_DRAFT_ADAPTER_MIB", "0")) << 20
    if not path:
        if budget:
            raise ValueError("inactive draft adapter cannot reserve weights")
        return
    identity = binding(tail)  # Service never admits a reference identity.
    model, manifest = load(path, identity, max_bytes=budget)
    install(tail, model, identity)
    tail._draft_adapter_status["weights_sha256"] = manifest["weights_sha256"]


def public_config(directory=None):
    import os
    directory = directory if directory is not None else os.environ.get("V4_DRAFT_ADAPTER_DIR", "")
    budget = int(os.environ.get("V4_DRAFT_ADAPTER_MIB", "0")) << 20
    if not directory:
        return {"enabled": False}
    path = Path(directory) / "manifest.json"
    if path.stat().st_size > 65536:
        raise ValueError("oversized draft adapter manifest")
    raw = path.read_bytes()
    meta = json.loads(raw)
    validate_binding(meta["binding"])
    if type(meta.get("weights_bytes")) is not int or not 0 < meta["weights_bytes"] <= budget:
        raise ValueError("configured draft adapter exceeds its declared reservation")
    return {"enabled": True, "schema": SCHEMA, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "weights_sha256": meta["weights_sha256"], "binding_sha256": digest(meta["binding"]),
            "device_reserved_bytes": budget, "host_reserved_bytes": 3 * budget,
            "release_qualified": False}
