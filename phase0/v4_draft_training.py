"""Bounded offline draft-only residual distillation and vocabulary audit.

Frozen safetensors are local teacher-forced exports, never remote pickle or
automatic customer-prompt capture. Training is CPU-only; collection uses only
the explicitly selected owned device and never rents or downloads hardware.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "engines" / "deepseek_v4"):
    sys.path.insert(0,str(directory))


def read(path, limit):
    path=Path(path)
    if path.stat().st_size > limit:
        raise ValueError("offline draft metadata exceeds declared bound")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    t=sub.add_parser("train")
    t.add_argument("--input",required=True)
    t.add_argument("--binding",required=True)
    t.add_argument("--out-dir",required=True)
    t.add_argument("--host-mib",type=int,required=True)
    t.add_argument("--rank",type=int,default=8)
    t.add_argument("--steps",type=int,default=100)
    t.add_argument("--batch-size",type=int,default=16)
    t.add_argument("--lr",type=float,default=.01)
    t.add_argument("--seed",type=int,default=0)
    a=sub.add_parser("audit")
    a.add_argument("--decoded-tokens",required=True,help="JSON object from token id to actual decoded text")
    a.add_argument("--proposed-ids",required=True,help="JSON original-token ID subset, not a foreign-model vocabulary")
    a.add_argument("--target-ids",help="JSON held-out workload target token IDs")
    a.add_argument("--out",required=True)
    c=sub.add_parser("collect")
    source=c.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--fixture",action="store_true")
    c.add_argument("--inputs",nargs="+",required=True,help="one frozen teacher-forced conversation per file")
    c.add_argument("--out-dir",required=True)
    c.add_argument("--lo",type=int,default=0)
    c.add_argument("--hi",type=int,required=True)
    c.add_argument("--device",default="cpu",choices=("cpu","cuda:0"))
    c.add_argument("--max-tokens",type=int,default=256)
    c.add_argument("--host-mib",type=int,required=True)
    c.add_argument("--device-mib",type=int,required=True)
    args=parser.parse_args(argv)
    from v4_draft_adapter import train,export,coverage
    if args.action=="audit":
        tokens=read(args.decoded_tokens,16 << 20)
        result=coverage({int(k):v for k,v in tokens.items()},read(args.proposed_ids,4 << 20),
                        target_ids=read(args.target_ids,4 << 20) if args.target_ids else ())
        Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    elif args.action=="collect":
        if (not 4<=len(args.inputs)<=128 or not 1<=args.host_mib<=65536
                or not 1<=args.device_mib<=1048576 or not 8<=args.max_tokens<=16384):
            raise ValueError("bounded collector inputs required")
        if sum(Path(p).stat().st_size for p in args.inputs)>min(2 << 30,(args.host_mib << 20)//2):
            raise ValueError("teacher exports exceed host input quota")
        import os
        if args.fixture:
            if args.device!="cpu": raise ValueError("fixture is not CUDA evidence")
            os.environ["V4_KERNELS"]="cpu"
        if (os.environ.get("V4_DRAFT_ADAPTER_DIR") or any(os.environ.get(k,"0") not in ("", "0")
                for k in ("V4_DSPARK_CHAIN_GRAPH", "V4_NATIVE_LANES", "V4_NATIVE_BATCH", "V4_CUDA_GRAPH"))
                or os.environ.get("V4_EXPERT_PLACEMENT","gpu")!="gpu"):
            raise ValueError("collection requires an unadapted eager native drafter")
        from safetensors.torch import load_file,save_file
        import torch
        import v4_stage as V
        import v4_dspark_draft as D
        from v4_draft_adapter import collect,binding
        from phase0.v4_benchmark import source_identity
        frozen_source=source_identity()
        torch.set_num_threads(1)
        previous=torch.get_default_device()
        model_args=V.config(args.checkpoint) if not args.fixture else None
        if args.fixture:
            import v4_ref_cpu as R
            model_args=R.cpu_args(max_seq_len=args.max_tokens)
        model_args.max_seq_len=args.max_tokens
        if args.hi!=model_args.n_layers or not 0<=args.lo<args.hi:
            raise ValueError("draft collection owns the tail span")
        if not args.fixture:
            V.verify_checkpoint_artifacts(args.checkpoint,required=True,lo=args.lo,hi=args.hi,
                head=args.lo==0,tail=True,dspark=True)
            from v4_resources import inspect_checkpoint,stage_storage_inventory
            inventory=stage_storage_inventory(inspect_checkpoint(args.checkpoint),args.lo,args.hi,
                head=args.lo==0,tail=True,dspark=True)
            if sum(t["runtime_parameter_floor_bytes"] for t in inventory["tensors"].values()) > args.device_mib << 20:
                raise ValueError("native checkpoint exceeds collection device quota before loading")
        cache_names=("precompute_freqs_cis","get_window_topk_idxs","get_compress_topk_idxs","get_dspark_topk_idxs")
        for name in cache_names: getattr(V.ref(),name).cache_clear()
        try:
            torch.set_default_device(args.device)
            stage=V.Stage(args.lo,args.hi,model_args,head=args.lo==0,tail=True,dspark=True,device=args.device)
            tail=D.DSparkTail(stage)
            if args.fixture:
                oracle=R.build_oracle(model_args,7)
                for index,layer in enumerate(stage.layers,args.lo): layer.load_state_dict(oracle.layers[index].state_dict())
                if stage.head: stage.embed_tokens.load_state_dict(oracle.embed.state_dict())
                stage.norm.load_state_dict(oracle.norm.state_dict());stage.lm_head.load_state_dict(oracle.head.state_dict())
                with torch.no_grad():
                    for name in ("hc_head_fn","hc_head_base","hc_head_scale"): getattr(stage,name).copy_(getattr(oracle,name))
                for index,block in enumerate(tail.mtp):
                    block.load_state_dict({k:v for k,v in oracle.mtp[index].state_dict().items() if k not in D.ALIAS_KEYS},strict=False)
                del oracle
            else:
                stage.load(args.checkpoint);tail.load(args.checkpoint)
            from v4_resources import measure_stage_resources
            storage=measure_stage_resources(stage,draft=tail,checkpoint_id="owned-draft-collection")["module_storage"]
            if storage["host_bytes"] > args.host_mib << 20 or storage["gpu_bytes"] > args.device_mib << 20:
                raise ValueError("loaded collection Stage exceeds owned quota")
            identity=binding(tail,reference=args.fixture,fixture_tokenizer="a"*64 if args.fixture else None)
            data=collect(tail,[load_file(p,device="cpu") for p in args.inputs],host_bytes=args.host_mib << 20)
            if source_identity()["sha256"] != frozen_source["sha256"]:
                raise ValueError("collection source changed during the owned run")
            directory=Path(args.out_dir);directory.mkdir(parents=True,exist_ok=False)
            save_file(data,str(directory/"training.safetensors"))
            (directory/"binding.json").write_text(json.dumps(identity,indent=2)+"\n",encoding="utf-8")
            provenance={"schema":"v4-draft-collection/1","binding":identity,"source_sha256":frozen_source["sha256"],
                "inputs":[{"sha256":hashlib.sha256(Path(p).read_bytes()).hexdigest(),"bytes":Path(p).stat().st_size} for p in args.inputs],
                "training_sha256":hashlib.sha256((directory/"training.safetensors").read_bytes()).hexdigest(),
                "rows":data["features"].shape[0],"device":args.device,"scope":"native_first_draft_slot_teacher_forced_exports",
                "release_qualified":False,"teacher_execution_proven":False}
            (directory/"collection-report.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
        finally:
            for name in cache_names: getattr(V.ref(),name).cache_clear()
            torch.set_default_device(previous)
    else:
        if not 1 <= args.host_mib <= 65536 or Path(args.input).stat().st_size > min(2 << 30,(args.host_mib << 20)//2):
            raise ValueError("frozen training input exceeds explicit host bound")
        from safetensors.torch import load_file
        import torch
        torch.set_num_threads(1)
        model,result=train(load_file(args.input,device="cpu"),read(args.binding,65536),
            rank=args.rank,steps=args.steps,batch_size=args.batch_size,lr=args.lr,seed=args.seed,
            host_bytes=args.host_mib << 20)
        export(args.out_dir,model,result)
    print(json.dumps({"action":args.action,"release_qualified":False,"gpu_qualified":False}))
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(f"draft training refused: {error}",file=sys.stderr)
        raise SystemExit(2)
