"""Real owned native fixture calibration and explicit serial wall partitions."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import os

import pytest
import torch

from shard.runtime_profile import RuntimeProfiler, validate_profile
from shard.stage_cost import surface_from_frames


def test_non_overlapping_wall_partitions_export_fixed_costs_and_drop_failed_frames():
    tick=[0.]
    def clock(): tick[0]+=.001; return tick[0]
    p=RuntimeProfiler(clock=clock,role_cost_capacity=5)
    meta=dict(decode_mode="greedy",role={"head":True,"tail":True,"draft":False},
        layer_start=0,layer_end=2,context_tokens=8,frame_tokens=1,warmness="warm",layer_weights=[1.,2.],completion=lambda:None)
    for _ in range(5):
        with p.cost_frame(**meta) as frame:
            with frame.phase("embedding"): pass
            with frame.phase("main"): pass
            with frame.phase("lm_head"): pass
    snapshot=p.snapshot()
    surface=surface_from_frames(snapshot["role_cost_frames"]["frames"],source_sha256="a"*64)
    assert surface["schema"]=="shard-stage-cost-surface/2"
    row=surface["samples"][0]
    assert all(sum(row["component_samples_ms"][k][i] for k in row["component_samples_ms"])==pytest.approx(total)
               for i,total in enumerate(row["frame_samples_ms"]))
    with pytest.raises(RuntimeError):
        with p.cost_frame(**meta) as frame:
            with frame.phase("main"): raise RuntimeError("failed native forward")
    assert len(p.snapshot()["role_cost_frames"]["frames"])==5
    with pytest.raises(ValueError,match="non-overlapping"):
        with p.cost_frame(**meta) as frame:
            with frame.phase("main"):
                with frame.phase("draft"): pass


def test_flat_worker_import_retains_role_frame_recording_and_signature_validation(tmp_path):
    root=Path(__file__).resolve().parents[1]
    code="""
import sys
sys.path.insert(0,'shard')
from runtime_profile import RuntimeProfiler,validate_profile
t=[0.]
def clock(): t[0]+=.001; return t[0]
p=RuntimeProfiler(clock=clock,role_cost_capacity=5)
for _ in range(5):
    with p.cost_frame(decode_mode='greedy',role={'head':False,'tail':False,'draft':False},
            layer_start=1,layer_end=2,context_tokens=8,frame_tokens=1,warmness='warm',
            layer_weights=[1.],completion=lambda:None) as frame:
        with frame.phase('main'): pass
assert len(validate_profile(p.snapshot())['role_cost_frames']['frames'])==5
"""
    result=subprocess.run([sys.executable,"-c",code],cwd=root,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize("mode,width",[("greedy",1),("dspark",2)])
def test_owned_calibration_cli_uses_real_native_fixture_and_loaded_draft(tmp_path,mode,width):
    from safetensors.torch import save_file
    frozen=tmp_path/"frozen.safetensors"; out=tmp_path/"cost.json"
    save_file({"token_ids":torch.arange(1,6+width).reshape(1,-1)},str(frozen))
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,"PYTHONUTF8":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "V4_KERNELS":"cpu","V4_NATIVE_LANES":"0","V4_NATIVE_BATCH":"0","V4_CUDA_GRAPH":"0"}
    result=subprocess.run([sys.executable,str(root/"phase0/v4_role_calibration.py"),"--fixture","--input",str(frozen),
        "--out",str(out),"--hi","4","--context","5","--frame-tokens",str(width),"--mode",mode,
        "--max-seq","64","--host-mib","128","--device-mib","128"],env=env,cwd=root,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    report=json.loads(out.read_text())
    assert report["release_qualified"] is False
    assert report["hardware"]["identity_complete"] is False
    assert report["cost_surface"]["decode_mode"]==mode
    assert report["cost_surface"]["role"]["draft"] is (mode=="dspark")
    sample=report["cost_surface"]["samples"][0]
    assert len(sample["frame_samples_ms"])==5
    assert all(x>0 for x in sample["component_samples_ms"]["main"])
    assert all(x>0 for x in sample["component_samples_ms"]["embedding"])
    if mode=="dspark": assert all(x>0 for x in sample["component_samples_ms"]["draft"])


def test_role_cost_export_traverses_signed_receipts_without_mixing_runtime_or_span():
    import base64
    from phase0.v4_profile_report import explain, role_surfaces
    from shard.receipt import _canonical
    # Build a signed synthetic campaign explicitly; this exercises the verifier
    # boundary without claiming any GPU hardware measurement.
    from test_v4_benchmark import observed_protocol, observed_report
    protocol,keys=observed_protocol(); report=observed_report(protocol,keys)
    for sample in report["samples"]:
        for index,receipt in enumerate(sample["receipts"]):
            lo,hi=receipt["layer_start"],receipt["layer_end"]
            p=RuntimeProfiler(role_cost_capacity=5)
            for _ in range(5):
                with p.cost_frame(decode_mode="greedy",role={"head":lo==0,"tail":hi==43,"draft":False},
                    layer_start=lo,layer_end=hi,context_tokens=8,frame_tokens=1,warmness="warm",
                    layer_weights=[1.]*(hi-lo),completion=lambda:None) as frame:
                    with frame.phase("main"): sum(range(100))
            from shard.runtime_metrics import RuntimeMetrics
            metrics=RuntimeMetrics("cuda:0").snapshot(); metrics["performance"]=p.snapshot()
            receipt["runtime_metrics"]=metrics
            receipt["sig"]=base64.b64encode(keys[index].sign(_canonical(receipt))).decode()
    result=role_surfaces(explain(report,protocol))
    assert result["surfaces"] and not result["release_qualified"]
    for row in result["surfaces"]:
        assert row["qualification"]=="diagnostic_input_requires_matching_runtime_resource_calibration"
        assert row["cost_surface"]["schema"]=="shard-stage-cost-surface/2"
        assert row["receipt_sha256"]
