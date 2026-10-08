import ast
import json
import shlex
from types import SimpleNamespace

import pytest
from deploy_oss import RingDeployment, ssh_argv
from shard.pipeline_plan import build_plan


def plan():
    return build_plan({'num_hidden_layers':36}, ring_id='test-ring', cohort_id='a'*64,
                      endpoints=['127.0.0.1:30001','127.0.0.1:30002','127.0.0.1:30003'],split='20,8,8')


def nodes():
    return {f'stage-{i}':{'ssh_target':f'node{i}','workspace':'/root/FlyAI','model':'/models/oss',
                          'node_key':'/root/receipt.key'} for i in range(3)}


def test_tail_first_owned_launch_and_manifest_on_windows_controller(tmp_path):
    calls = []
    def execute(argv, **kwargs):
        code = shlex.split(argv[-1])[-1]
        ast.parse(code)
        calls.append(code)
        result = {'running':True,'reused':False,'listening':True}
        return SimpleNamespace(returncode=0,stdout=json.dumps(result),stderr='')
    deploy = RingDeployment(plan(), nodes(), state_dir=tmp_path, executor=execute)
    result = deploy.deploy(warmup=False)
    starts = [s for s in calls if 'm.start(' in s]
    assert ['test-ring.stage'+str(i) in s for i,s in zip([2,1,0],starts)] == [True]*3
    assert all('/root/FlyAI/.shard-deployments/test-ring/plan.json' in s for s in starts)
    assert all('kill -9' not in s and 'fuser' not in s for s in calls)
    assert result['signed_warmup'] is False


def test_failure_only_rolls_back_processes_started_by_this_attempt(tmp_path):
    calls=[]
    def execute(argv, **kwargs):
        code = shlex.split(argv[-1])[-1]; calls.append(code)
        if 'm.start(' in code:
            if 'test-ring.stage1' in code: return SimpleNamespace(returncode=1,stdout='',stderr='private')
            return SimpleNamespace(returncode=0,stdout=json.dumps({'running':True,'reused':False}),stderr='')
        return SimpleNamespace(returncode=0,stdout=json.dumps({'running':True,'listening':True}),stderr='')
    deploy=RingDeployment(plan(),nodes(),state_dir=tmp_path,executor=execute)
    with pytest.raises(RuntimeError): deploy.deploy(warmup=False)
    stops=[s for s in calls if '.stop(' in s]
    assert len(stops)==1 and 'test-ring.stage2' in stops[0]


def test_forward_is_separate_argv_and_exit_failure_is_required():
    argv=ssh_argv({'ssh_target':'node','ssh_port':22},forward='127.0.0.1:30001:localhost:29501')
    assert '-L' in argv and '-R' not in argv
    assert 'ExitOnForwardFailure=yes' in argv
    with pytest.raises(ValueError): ssh_argv({'ssh_target':'-oProxyCommand=bad'})


def test_private_environment_uses_stdin_not_ssh_or_remote_command_argv(tmp_path):
    entries=nodes(); entries['stage-2']['environment']={'HF_TOKEN':'private-placeholder-test'}
    captured=[]
    def execute(argv,**kwargs):
        assert 'private-placeholder-test' not in ' '.join(argv)
        captured.append(kwargs.get('input',''))
        return SimpleNamespace(returncode=0,stdout=json.dumps({'running':True,'reused':False,'listening':True}),stderr='')
    RingDeployment(plan(),entries,state_dir=tmp_path,executor=execute).deploy(warmup=False)
    assert any('private-placeholder-test' in data for data in captured)
