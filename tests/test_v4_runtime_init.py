"""Runtime contracts on CPU and optional real CUDA probes; no performance claims."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

import v4_runtime_init as INIT
import v4_kernels_cpu as CPU
import v4_ref_cpu as REF
import v4_stage as V4
import v4_whole_layer_graph as WL


@pytest.fixture
def clean_hadamard(monkeypatch):
    monkeypatch.setattr(INIT, "_FUNCTION", None)
    monkeypatch.setattr(INIT, "_IDENTITY", None)
    monkeypatch.setattr(INIT, "_PROBES", {})
    monkeypatch.setattr(INIT, "_FAILED", {})
    monkeypatch.setattr(INIT, "V4_HADAMARD", "torch")
    monkeypatch.delitem(sys.modules, "fast_hadamard_transform", raising=False)
    yield


def test_shared_torch_hadamard_uses_actual_shapes_and_stable_identity(clean_hadamard):
    assert INIT.hadamard_identity()["backend"] == "uninitialized"
    original_kernel = sys.modules.get("kernel")
    assert INIT.ensure_hadamard() == "torch"
    identity = INIT.hadamard_identity()
    assert sys.modules.get("kernel") is original_kernel
    assert INIT.probe_hadamard("cpu", torch.bfloat16, (1, 1, 4, 128))["passed"]
    assert INIT.probe_hadamard("cpu", torch.bfloat16, (1, 1, 4, 128))["passed"]
    assert INIT.probe_hadamard("cpu", torch.bfloat16, (1, 1, 128))["passed"]
    assert INIT.hadamard_status()["probe_count"] == 2
    assert INIT.hadamard_identity() == identity, "observations must not change calibration identity"
    json.dumps(INIT.hadamard_status(), allow_nan=False)


def test_tilelang_install_keeps_real_kernel_and_uses_shared_hadamard(clean_hadamard, monkeypatch):
    sentinel = types.ModuleType("kernel")
    monkeypatch.setitem(sys.modules, "kernel", sentinel)
    monkeypatch.setattr(CPU, "V4_KERNELS", "tilelang")
    assert CPU.install() == "tilelang"
    assert sys.modules["kernel"] is sentinel
    assert INIT.hadamard_identity()["backend"] == "torch"


def test_importable_extension_with_broken_runtime_is_refused_before_token_zero(clean_hadamard, monkeypatch):
    module = types.ModuleType("fast_hadamard_transform")
    def broken(*args, **kwargs):
        raise RuntimeError("device function unavailable")
    module.hadamard_transform = broken
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(INIT, "V4_HADAMARD", "extension")
    assert INIT.ensure_hadamard() == "extension"
    with pytest.raises(RuntimeError, match="before READY"):
        INIT.probe_hadamard("cpu", torch.bfloat16, (1, 1, 128))
    assert INIT.hadamard_identity()["backend"] == "extension"
    assert INIT.hadamard_status()["failed_probes"][0]["error_type"] == "RuntimeError"
    with pytest.raises(RuntimeError, match="already refused"):
        INIT.probe_hadamard("cpu", torch.bfloat16, (1, 1, 128))


def test_backend_replacement_cannot_silently_change_live_math(clean_hadamard, monkeypatch):
    INIT.ensure_hadamard()
    module = types.ModuleType("fast_hadamard_transform")
    module.hadamard_transform = lambda x, scale=1.: x
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="changed after initialization"):
        INIT.ensure_hadamard()


def test_bad_requested_hadamard_mode_is_not_auto_fallback(clean_hadamard, monkeypatch):
    monkeypatch.setattr(INIT, "V4_HADAMARD", "unknown")
    with pytest.raises(ValueError, match="auto, extension or torch"):
        INIT.ensure_hadamard()


def test_field_plain_import_shim_is_torch_reference_not_native_extension(clean_hadamard, monkeypatch):
    shim = types.ModuleType("fast_hadamard_transform")
    shim.hadamard_transform = CPU.hadamard_transform
    monkeypatch.setitem(sys.modules, shim.__name__, shim)
    monkeypatch.setattr(INIT, "V4_HADAMARD", "auto")
    assert INIT.ensure_hadamard() == "torch"
    assert INIT.hadamard_identity()["backend"] == "torch"


def block():
    args = REF.cpu_args()
    model = V4.ref()
    V4._set_globals(model, args)
    with model.set_dtype(torch.bfloat16):
        return model.Block(2, args), args, model


def test_explicit_alias_binding_precedes_any_eager_forward_and_preserves_data():
    layer, _, _ = block()
    attn = layer.attn
    assert attn.compressor.kv_cache is None and attn.indexer.compressor.kv_cache is None
    original = attn.kv_cache.clone()
    bound = INIT.bind_attention_aliases(attn)
    assert bound["changed"]
    assert attn.compressor.freqs_cis is attn.freqs_cis
    assert attn.indexer.freqs_cis is attn.freqs_cis
    assert attn.indexer.compressor.freqs_cis is attn.freqs_cis
    assert attn.indexer.compressor.kv_cache is attn.indexer.kv_cache
    assert attn.compressor.kv_cache.untyped_storage().data_ptr() == attn.kv_cache.untyped_storage().data_ptr()
    assert torch.equal(attn.kv_cache, original)
    assert not INIT.bind_attention_aliases(attn)["changed"]
    attn.compressor.kv_cache = attn.compressor.kv_cache.clone()
    with pytest.raises(RuntimeError, match="owning buffer"):
        INIT.bind_attention_aliases(attn)


def test_buffer_replacement_retires_all_graphs_and_rebinds_real_storage(monkeypatch):
    layer, args, model = block()
    stage = types.SimpleNamespace(args=args, device="cpu", dtype=torch.bfloat16, _M=model)
    graphs = WL.WholeBlockGraphs(layer, stage)
    graphs._graphs[(32, False)] = {"graph": object()}
    graphs.g_post = {"graph": object()}
    graphs._pool = object(); graphs.ho = [object()]; graphs.ffn_out_buf = object()
    monkeypatch.setattr(WL, "_GRAPH_COUNT", 2)
    layer.attn.kv_cache = layer.attn.kv_cache.clone()
    graphs._check_aliases()
    assert not graphs._graphs and graphs.g_post is None and graphs._pool is None
    assert graphs.ho is None and graphs.ffn_out_buf is None and WL._GRAPH_COUNT == 0
    assert graphs.alias_invalidations == 1
    assert INIT.tensor_storage_signature(layer.attn.compressor.kv_cache) == INIT.tensor_storage_signature(
        layer.attn.kv_cache[:, layer.attn.window_size:])
    graphs._check_aliases()
    assert graphs.alias_invalidations == 1


@pytest.mark.parametrize("indexer,name", [(False, "kv_state"), (False, "score_state"),
                                       (True, "kv_state"), (True, "score_state")])
def test_compressor_recurrence_buffer_replacement_also_invalidates_graphs(monkeypatch, indexer, name):
    layer, args, model = block()
    stage = types.SimpleNamespace(args=args, device="cpu", dtype=torch.bfloat16, _M=model)
    graphs = WL.WholeBlockGraphs(layer, stage)
    graphs._graphs[(32, False)] = {"graph": object()}
    monkeypatch.setattr(WL, "_GRAPH_COUNT", 1)
    compressor = layer.attn.indexer.compressor if indexer else layer.attn.compressor
    setattr(compressor, name, getattr(compressor, name).clone())
    graphs._check_aliases()
    assert not graphs._graphs and graphs.alias_invalidations == 1 and WL._GRAPH_COUNT == 0


def test_stage_aliases_survive_in_place_reset_and_chunk_scratch_replacement():
    args = REF.cpu_args()
    stage = V4.Stage(2, 3, args, device="cpu", fast_verify=True)
    attn = stage.layers[0].attn
    signature = INIT.bind_attention_aliases(attn)["signature"]
    stage.reset()
    assert INIT.bind_attention_aliases(attn)["signature"] == signature
    old_pointer = attn.kv_cache.untyped_storage().data_ptr()
    stage._reserve_chunk_scratch()
    assert attn.kv_cache.untyped_storage().data_ptr() != old_pointer
    assert not INIT.bind_attention_aliases(attn)["changed"]


def test_cpu_selftest_prepares_before_first_reference_import_even_on_cuda_host(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = ("import sys,torch;sys.path.insert(0," + repr(str(root / 'engines/deepseek_v4')) + ");"
            "torch.cuda.is_available=lambda:True;from v4_runtime_init import prepare_cpu_selftest;"
            "prepare_cpu_selftest();import v4_ref_cpu,v4_kernels_cpu;"
            "a=v4_ref_cpu.cpu_args();m=v4_ref_cpu.build_oracle(a);"
            "assert v4_kernels_cpu.backend()=='cpu';"
            "assert all(p.device.type=='cpu' for p in m.parameters());print('CPU_ISOLATED')")
    env = dict(os.environ, V4_KERNELS="auto"); env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and "CPU_ISOLATED" in result.stdout, result.stderr


def test_cpu_selftest_refuses_cached_gpu_kernel_without_replacing_it(clean_hadamard, monkeypatch):
    sentinel = types.ModuleType("kernel")
    monkeypatch.setitem(sys.modules, "kernel", sentinel)
    with pytest.raises(RuntimeError, match="fresh selftest process"):
        INIT.prepare_cpu_selftest()
    assert sys.modules["kernel"] is sentinel


@pytest.mark.parametrize("command", ["selftest", "selftest-relay"])
def test_fresh_cpu_selftest_ignores_gpu_launcher_recipe_without_claiming_it(tmp_path, command):
    import v4_benchmark
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, **v4_benchmark.DEFAULT_ENV, "V4_LEVERS_STRICT": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env.pop("PYTHONPATH", None)
    try:
        result = subprocess.run([sys.executable, str(root / "engines/deepseek_v4/v4_pipe.py"), command],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or b"") + (error.stderr or b"")
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        pytest.fail("fresh CPU selftest timed out:\n" + output[-8000:])
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "CPU_SELFTEST_RECIPE" in result.stdout and "ALL PASS" in result.stdout
    assert '"V4_CUDA_GRAPH": "0"' in result.stdout and '"V4_LEVERS_STRICT": "0"' in result.stdout


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA Hadamard smoke unavailable")
def test_actual_cuda_hadamard_smoke_and_graph_capture(clean_hadamard):
    INIT.probe_hadamard("cuda:0", torch.bfloat16, (1, 1, 64, 128))
    function = sys.modules["fast_hadamard_transform"].hadamard_transform
    x = torch.randn(1, 1, 64, 128, device="cuda", dtype=torch.bfloat16)
    for _ in range(3): function(x, scale=128 ** -.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function(x, scale=128 ** -.5)
    graph.replay()
    assert torch.equal(output, CPU.hadamard_transform(x, scale=128 ** -.5))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA graph recapture unavailable")
@pytest.mark.skipif(CPU.backend() != "cpu", reason="tiny BF16 fixture requires Torch reference kernels")
def test_actual_cuda_buffer_replacement_recaptures_compute_and_state(clean_hadamard, monkeypatch):
    """Real CUDA replay versus the eager twin, before and after pointer replacement."""
    previous_device = torch.get_default_device()
    monkeypatch.setattr(WL, "_GRAPH_COUNT", 0)
    try:
        torch.set_default_device("cuda")
        layer, args, model = block()
        REF.init_random(layer, 7)
        stage = types.SimpleNamespace(args=args, device="cuda", dtype=torch.bfloat16, _M=model)
        graphs = WL.WholeBlockGraphs(layer, stage, moe_mode="eager")
        with torch.no_grad():
            ids = torch.arange(20, device="cuda").view(1, 20)
            h = torch.randn(1, 20, args.hc_mult, args.dim, device="cuda", dtype=torch.bfloat16)
            layer(h, 0, ids)
            x, tok = h[:, -1:].clone(), ids[:, -1:].clone()
            old_graph = None
            for step, position in enumerate((20, 23)):
                if step:
                    layer.attn.kv_cache = layer.attn.kv_cache.clone()
                    layer.attn.compressor.kv_state = layer.attn.compressor.kv_state.clone()
                    layer.attn.indexer.compressor.score_state = layer.attn.indexer.compressor.score_state.clone()
                before = [tensor.clone() for tensor in WL._layer_state(layer)]
                got = graphs.run(x, tok, position).clone()
                after = [tensor.clone() for tensor in WL._layer_state(layer)]
                assert graphs._graphs and not graphs.eager, "fixture must really capture and replay"
                current_graph = next(iter(graphs._graphs.values()))["graph"]
                if step:
                    assert graphs.alias_invalidations == 1 and current_graph is not old_graph
                old_graph = current_graph
                for tensor, saved in zip(WL._layer_state(layer), before):
                    tensor.copy_(saved)
                wanted = graphs._eager(x, tok, position).clone()
                assert torch.equal(got, wanted)
                for tensor, saved in zip(WL._layer_state(layer), after):
                    assert torch.equal(tensor, saved), "graph/eager recurrence and KV must agree"
    finally:
        torch.set_default_device(previous_device)
