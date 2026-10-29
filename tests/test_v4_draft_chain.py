"""Native MTP chain transcription against untouched independent scalar drafts."""
import pytest
from copy import deepcopy
import torch
import v4_ref_cpu as R
import v4_dspark_fast as F
from v4_draft_chain import DraftChain
from test_v4_dspark_fast import _build, _record


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("seed", [3, 7, 23])
def test_complete_chain_keeps_all_native_mtp_outputs_confidence_and_window_bytes(seed):
    args = R.cpu_args()
    oracle = R.build_oracle(args, 7)
    _, scalar = _build(oracle, args)
    _, actual = _build(oracle, args)
    token, hidden, rounds = _record(args, [1, 3, 2, 4, 1], seed)
    scalar.prefill(token, hidden); actual.prefill(token, hidden)
    chain = DraftChain(actual, graph_bytes=16 << 20, max_graphs=16, reference=True)
    for ids, mh, position in rounds:
        wanted, confidence = scalar.advance_and_draft(ids, mh, position)
        with torch.no_grad():
            outputs = chain.run(ids, mh, position)
        actual._pos = position + ids.shape[1] - 1
        assert torch.equal(outputs[0][:, 1:], wanted)
        assert torch.equal(outputs[1], scalar.last_spec[1])
        assert torch.equal(outputs[2], confidence)
        for a, b in zip(actual.mtp, scalar.mtp):
            assert torch.equal(a.attn.kv_cache.view(torch.uint8), b.attn.kv_cache.view(torch.uint8))
    assert chain.status()["captures"] == chain.status()["replays"] == 0
    assert chain.status()["reference_eager"] == len(rounds)


def test_service_fast_advance_uses_configured_reference_chain_and_preserves_cursors(monkeypatch):
    args = R.cpu_args()
    oracle = R.build_oracle(args, 7)
    _, scalar = _build(oracle, args)
    _, actual = _build(oracle, args)
    actual._draft_chain = DraftChain(actual, graph_bytes=16 << 20, reference=True)
    token, hidden, rounds = _record(args, [1, 3, 2], 7)
    scalar.prefill(token, hidden); actual.prefill(token, hidden)
    for ids, mh, position in rounds:
        wanted = scalar.advance_and_draft(ids, mh, position)
        got = F.fast_advance_and_draft(actual, ids, mh, position)
        assert all(torch.equal(a, b) for a, b in zip(wanted, got))
        assert actual.pos == scalar.pos
    with torch.no_grad():
        actual.mtp[0].attn.wkv.weight.add_(1)
    ids, mh, position = rounds[-1]
    with pytest.raises(RuntimeError, match="ownership/model"):
        actual._draft_chain.run(ids, mh, position)


def test_invalid_chain_cannot_allocate_or_masquerade_as_cuda():
    args = R.cpu_args()
    _, actual = _build(R.build_oracle(args, 7), args)
    with pytest.raises(ValueError, match="honest"):
        DraftChain(actual, graph_bytes=1 << 20)
    with pytest.raises(ValueError, match="backup"):
        DraftChain(actual, graph_bytes=1, reference=True)


def test_complete_chain_composes_with_native_packed_draft_pair_backend(monkeypatch):
    import v4_dspark_moe as moe
    for name in ("_MOD", "_WORLD_SIZE"):
        monkeypatch.setattr(moe,name,getattr(moe,name))
    monkeypatch.setattr(moe,"V4_DSPARK_MOE",True)
    args=R.cpu_args()
    oracle=R.build_oracle(args,7)
    _,scalar=_build(oracle,args);_,actual=_build(oracle,args)
    for tail in (scalar,actual):
        moe.swap_in_fp4_moes(tail,moe_inter_dim=128,seed=17)
    # Bind only the optimized drafter's instances. The reference class and main
    # verifier remain untouched, using their original native per-expert math.
    assert moe.install_drafter(actual)==len(actual.mtp)
    token,hidden,rounds=_record(args,[1,3,2],11)
    scalar.prefill(token,hidden);actual.prefill(token,hidden)
    chain=DraftChain(actual,graph_bytes=16 << 20,reference=True)
    for ids,mh,position in rounds:
        scalar.advance_and_draft(ids,mh,position)
        with torch.no_grad(): got=chain.run(ids,mh,position)
        actual._pos=position+ids.shape[1]-1
        assert all(torch.equal(a,b) for a,b in zip(got,scalar.last_spec))
        assert all(torch.equal(a.attn.kv_cache,b.attn.kv_cache) for a,b in zip(actual.mtp,scalar.mtp))
    assert all(steps==len(rounds) and not declined for steps,declined in moe.coverage(actual).values())
    assert chain._supports_capture()
    actual.block_size=1
    assert not chain._supports_capture()  # Native pair_forward would decline s=1.


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA draft-chain acceptance required")
def test_cuda_draft_chain_full_program_changes_inputs_and_restores_capture_state(monkeypatch):
    # The native FP4 checkpoint gate is separate. This synthetic dense FFN probe
    # exercises actual CUDA capture and live position/input updates without
    # substituting CPU evidence for a GPU capture.
    import v4_stage as V
    old = torch.get_default_device()
    names = ("precompute_freqs_cis", "get_dspark_topk_idxs")

    class Dense(torch.nn.Linear):
        def forward(self, value, ids):
            return super().forward(value)

    try:
        with torch.device("cpu"):
            args = R.cpu_args()
            oracle = R.build_oracle(args, 7)
            _, scalar = _build(oracle, args)
            _, actual = _build(oracle, args)
            token, hidden, rounds = _record(args, [1, 3, 2], 7)
            for index, (a, b) in enumerate(zip(actual.mtp, scalar.mtp)):
                with torch.random.fork_rng():
                    torch.manual_seed(index)
                    a.ffn = Dense(args.dim, args.dim, bias=False).to(actual.dtype)
                b.ffn = deepcopy(a.ffn)
        for tail in (actual, scalar):
            tail.device = tail.stage.device = "cuda"
            tail.mtp.cuda()
            for block in tail.mtp:
                block.attn.freqs_cis = block.attn.freqs_cis.cuda()
        for name in names:
            getattr(V.ref(), name).cache_clear()
        torch.set_default_device("cuda")
        scalar.prefill(token.cuda(), hidden.cuda()); actual.prefill(token.cuda(), hidden.cuda())
        chain = DraftChain(actual, graph_bytes=128 << 20, max_graphs=16)
        for ids, mh, position in rounds:
            wanted = scalar.advance_and_draft(ids.cuda(), mh.cuda(), position)
            with torch.no_grad():
                got = chain.run(ids.cuda(), mh.cuda(), position)
            actual._pos = position + ids.shape[1] - 1
            assert torch.equal(got[0][:, 1:], wanted[0])
            assert torch.equal(got[1], scalar.last_spec[1]) and torch.equal(got[2], wanted[1])
            assert all(torch.equal(a.attn.kv_cache, b.attn.kv_cache) for a,b in zip(actual.mtp,scalar.mtp))
        # Same graph keys, new request and different data, with a fresh reference.
        scalar.reset(); actual.reset()
        scalar.prefill(token.cuda(), hidden.cuda()); actual.prefill(token.cuda(), hidden.cuda())
        for ids, mh, position in rounds:
            ids = (ids.cuda() + 3) % args.vocab_size
            mh = mh.cuda() + .125
            wanted = scalar.advance_and_draft(ids, mh, position)
            with torch.no_grad():
                got = chain.run(ids, mh, position)
            actual._pos = position + ids.shape[1] - 1
            assert torch.equal(got[0][:, 1:], wanted[0])
            assert torch.equal(got[1], scalar.last_spec[1])
        torch.cuda.synchronize()
        assert chain.status()["captures"] > 0 and chain.status()["replays"] > chain.status()["captures"]
        assert chain.status()["capture_failures"] == 0
        # A real side-stream warmup failure occurs after native KV mutation.
        # Capture refusal must restore every byte before eager fallback may run.
        saved=[block.attn.kv_cache.clone() for block in actual.mtp]
        original=chain._program
        def failure_after_state(values):
            original(values)
            raise RuntimeError("owned warmup failure")
        monkeypatch.setattr(chain,"_program",failure_after_state)
        ids=rounds[-1][0][:,:1].cuda().expand(1,4).contiguous()
        mh=rounds[-1][1][:,:1].cuda().expand(1,4,-1).contiguous()
        with torch.no_grad():
            assert chain.run(ids,mh,32) is None
        torch.cuda.synchronize()
        assert all(torch.equal(block.attn.kv_cache,state) for block,state in zip(actual.mtp,saved))
        assert chain.status()["capture_failures"]==1 and not chain.status()["poisoned"]
    finally:
        for name in names:
            getattr(V.ref(), name).cache_clear()
        torch.set_default_device(old)
