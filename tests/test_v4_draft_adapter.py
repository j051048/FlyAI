from copy import deepcopy
import json
import pytest
import torch

from v4_draft_adapter import binding,coverage,train,export,load,install,Residual
import v4_ref_cpu as R
from test_v4_dspark_fast import _build, _record


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture():
    args=R.cpu_args()
    _,tail=_build(R.build_oracle(args,7),args)
    return tail,binding(tail,reference=True,fixture_tokenizer="a"*64)


def dataset(identity):
    n,dim,vocab=64,identity["hidden_dim"],identity["vocab_size"]
    x=torch.zeros(n,dim)
    labels=torch.arange(n)%4
    x[torch.arange(n),labels]=1
    return {"features":x,"base_logits":torch.zeros(n,vocab),"target_ids":labels.long(),
            "group_ids":(torch.arange(n)//4).long()}


def test_real_optimizer_improves_independent_conversation_holdout_and_roundtrips(tmp_path):
    _,identity=fixture()
    model,report=train(dataset(identity),identity,rank=4,steps=50,batch_size=16,lr=.05,host_bytes=64 << 20)
    assert report["heldout_loss_after"] < report["heldout_loss_before"]
    assert report["training_group_sha256"] != report["heldout_group_sha256"]
    assert not report["release_qualified"] and not report["teacher_execution_proven"]
    directory=tmp_path/"adapter"
    export(directory,model,report)
    restored,manifest=load(directory,identity,max_bytes=8 << 20)
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),restored.parameters()))
    wrong={**identity,"tokenizer_sha256":"b"*64}
    with pytest.raises(ValueError,match="identity"):
        load(directory,wrong,max_bytes=8 << 20)
    path=directory/"adapter.safetensors"
    data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(ValueError,match="weight bytes"):
        load(directory,identity,max_bytes=8 << 20)


def test_adapter_only_changes_draft_head_never_shared_target_weights_or_cache_frontier():
    tail,identity=fixture()
    target=[(p,p.detach().clone()) for module in tail.stage._owned_modules() for p in module.parameters()]
    native,reference=fixture()
    ids=torch.tensor([3])
    hidden=torch.randn(1,tail.block_size,tail.args.hc_mult,tail.args.dim).to(tail.dtype)
    base=tail.mtp[-1].forward_head(hidden,ids)
    model=Residual(tail.args.dim,tail.args.vocab_size,4)
    install(tail,model,identity)
    same=tail.mtp[-1].forward_head(hidden,ids)
    assert all(torch.equal(a,b) for a,b in zip(base,same))
    with torch.no_grad():
        model.up[7].fill_(100.)
    tail.mtp[-1].forward_head(hidden,ids)
    assert all(torch.equal(p,saved) for p,saved in target)
    assert tail.pos is None


def test_subsets_cannot_hide_chinese_or_observed_target_coverage():
    result=coverage({0:"code",1:"中文",2:"。",3:"hello"},[0,3],target_ids=[1,1,0])
    assert result["scripts"]["cjk"]=={"total":1,"covered":0}
    assert result["uncovered_targets"]==2
    assert not result["gpu_qualified"]
    with pytest.raises(ValueError,match="outside"):
        coverage({0:"a"},[99])


def test_training_refuses_quota_leakage_nonfinite_or_tokenizer_mismatch():
    _,identity=fixture(); data=dataset(identity)
    with pytest.raises(ValueError,match="quota"):
        train(data,identity,host_bytes=1)
    data["group_ids"].zero_()
    with pytest.raises(ValueError,match="four conversation"):
        train(data,identity,host_bytes=64 << 20)
    data=dataset(identity);data["features"][0,0]=float("nan")
    with pytest.raises(ValueError,match="nonfinite"):
        train(data,identity,host_bytes=64 << 20)
    tail,correct=fixture()
    with pytest.raises(ValueError,match="another native"):
        install(tail,Residual(tail.args.dim,tail.args.vocab_size,4),{**correct,"checkpoint_sha256":"f"*64})


def test_changed_draft_proposals_still_emit_only_independently_verified_greedy_tokens(monkeypatch):
    import test_v4_dspark as controls
    args=R.cpu_args()
    prompt,wanted=controls._rich_ids(args,controls.GEN+args.dspark_block_size)
    original=controls.prefilled
    def adapted(*args,**kwargs):
        stage,tail,token=original(*args,**kwargs)
        identity=binding(tail,reference=True,fixture_tokenizer="a"*64)
        model=Residual(tail.args.dim,tail.args.vocab_size,4,seed=19)
        with torch.no_grad():
            model.up.fill_(3.)
            model.up[7].fill_(100.)
        install(tail,model,identity)
        return stage,tail,token
    monkeypatch.setattr(controls,"prefilled",adapted)
    got,drafted,rewound=controls._spec_loop(R.build_oracle(args,controls.SEED),args,prompt,lambda values:values)
    assert got==wanted[:len(got)] and drafted>=1 and rewound>=1


def test_native_catalogue_binding_preserves_logical_hash_namespace(monkeypatch):
    from shard import weight_artifacts
    tail,_=fixture()
    tail.stage._artifact_verification={"checkpoint_id":"tensor-sha256:"+"c"*64,
        "catalog":{"assets":{"tokenizer.json":{"sha256":"d"*64}}}}
    checked=[]
    monkeypatch.setattr(weight_artifacts,"verified_stage_artifact_descriptor",lambda p:checked.append(p))
    from v4_draft_adapter import validate_binding
    identity=validate_binding(binding(tail))
    assert identity["checkpoint_sha256"]=="c"*64 and checked and not identity["reference"]


def teacher_records(tail):
    g=torch.Generator().manual_seed(11)
    return [{"prompt_taps":torch.randn(1,3,tail.hidden_dim,generator=g).to(tail.dtype),
        "main_taps":torch.randn(1,2,tail.hidden_dim,generator=g).to(tail.dtype),
        "anchor_ids":torch.randint(tail.args.vocab_size,(2,),generator=g),
        "teacher_logits":torch.randn(2,tail.args.vocab_size,generator=g)} for _ in range(4)]


def test_collection_runs_original_mtp_and_retains_conversation_groups_and_native_logits():
    from v4_draft_adapter import collect
    tail,_=fixture(); records=teacher_records(tail)
    control,_=fixture()
    item=records[0]
    control.prefill(item["anchor_ids"][:1],item["prompt_taps"])
    with torch.no_grad():
        control.advance_and_draft(item["anchor_ids"][:1].reshape(1,1),item["main_taps"][:,:1],3)
    expected=control.last_spec[1][:,0].clone()
    data=collect(tail,records,host_bytes=64 << 20)
    assert tail.pos is None and torch.equal(data["base_logits"][:1],expected)
    assert data["features"].shape==(8,tail.args.dim)
    assert data["group_ids"].tolist()==[0,0,1,1,2,2,3,3]
    assert torch.equal(data["target_ids"][:2],item["teacher_logits"].argmax(-1))
    broken=deepcopy(records);broken[0]["prompt_taps"]=None
    with pytest.raises(ValueError,match="layout"):
        collect(tail,broken,host_bytes=64 << 20)
    with pytest.raises(ValueError,match="quota"):
        collect(tail,records,host_bytes=1)


def test_offline_collector_cli_produces_bounded_safetensors_for_real_training(tmp_path):
    import subprocess,sys,os
    from pathlib import Path
    from safetensors.torch import save_file,load_file
    tail,_=fixture(); files=[]
    for index,record in enumerate(teacher_records(tail)):
        path=tmp_path/f"conversation-{index}.safetensors"
        save_file(record,str(path));files.append(str(path))
    directory=tmp_path/"collected"
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,"PYTHONUTF8":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "V4_KERNELS":"cpu","V4_CUDA_GRAPH":"0","V4_NATIVE_LANES":"0","V4_NATIVE_BATCH":"0",
        "V4_DSPARK_CHAIN_GRAPH":"0","V4_EXPERT_PLACEMENT":"gpu","V4_DRAFT_ADAPTER_DIR":""}
    result=subprocess.run([sys.executable,str(root/"phase0/v4_draft_training.py"),"collect",
        "--fixture","--inputs",*files,"--out-dir",str(directory),"--hi","8",
        "--host-mib","256","--device-mib","256"],env=env,cwd=root,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    data=load_file(str(directory/"training.safetensors"))
    identity=json.loads((directory/"binding.json").read_text())
    _,report=train(data,identity,rank=4,steps=2,batch_size=4,host_bytes=64 << 20)
    assert report["training_rows"]==6 and report["heldout_rows"]==2 and not report["release_qualified"]
