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


def test_warmup_runs_on_declared_coordinator_and_copies_its_bound_plan(tmp_path):
    contract = plan()
    contract['coordinator']['node_id'] = 'entry'
    entries = nodes()
    entries['entry'] = {'ssh_target': 'entry-host', 'workspace': '/entry/FlyAI',
        'model': '/models/oss', 'node_key': '/entry/node.key',
        'coordinator_key': '/entry/controller.key', 'transport': 'tcp', 'io_timeout_s': 777}
    warmups = []
    def execute(argv, **kwargs):
        code = shlex.split(argv[-1])[-1]
        ast.parse(code)
        if 'r=subprocess.run(' in code:
            warmups.append((argv, code, json.loads(kwargs['input'])))
            return SimpleNamespace(returncode=0, stdout='RESULT ' + json.dumps(
                {'proof_verified': True, 'output_ids': [7, 8]}), stderr='')
        return SimpleNamespace(returncode=0, stdout=json.dumps(
            {'running': True, 'reused': False, 'listening': True}), stderr='')
    RingDeployment(contract, entries, state_dir=tmp_path, executor=execute).deploy()
    argv, code, payload = warmups[0]
    assert argv[-2] == 'entry-host'
    assert '/entry/FlyAI/.shard-deployments/test-ring/plan.json' in code
    assert '/entry/controller.key' in code and "'--timeout', '777'" in code
    assert payload['plan'] == contract


def test_mapped_endpoint_can_use_distinct_container_listener(tmp_path):
    contract = plan()
    entry = nodes()['stage-0']
    entry['listen_port'] = 29501
    deployment = RingDeployment(contract, nodes(), state_dir=tmp_path)
    argv, _ = deployment.stage_command(contract['stages'][0], entry)
    assert argv[argv.index('--listen-port') + 1] == '29501'
    entry['listen_port'] = '29501'
    with pytest.raises(ValueError, match='listen_port'):
        deployment.stage_command(contract['stages'][0], entry)


def test_unknown_coordinator_is_rejected_before_any_remote_operation(tmp_path):
    contract = plan()
    contract['coordinator']['node_id'] = 'missing'
    with pytest.raises(ValueError, match='coordinator'):
        RingDeployment(contract, nodes(), state_dir=tmp_path)


def test_no_warmup_still_installs_the_independent_coordinator_plan(tmp_path):
    contract = plan()
    contract['coordinator']['node_id'] = 'entry'
    entries = nodes()
    entries['entry'] = {**entries['stage-0'], 'ssh_target': 'entry-host', 'workspace': '/entry/FlyAI'}
    calls = []
    def execute(argv, **kwargs):
        code = shlex.split(argv[-1])[-1]
        calls.append((argv, code, kwargs))
        assert 'r=subprocess.run(' not in code
        return SimpleNamespace(returncode=0, stdout=json.dumps(
            {'running': True, 'reused': False, 'listening': True}), stderr='')
    result = RingDeployment(contract, entries, state_dir=tmp_path, executor=execute).deploy(warmup=False)
    entry_calls = [call for call in calls if call[0][-2] == 'entry-host']
    assert len(entry_calls) == 1
    assert '/entry/FlyAI/.shard-deployments/test-ring/plan.json' in entry_calls[0][1]
    assert json.loads(entry_calls[0][2]['input'])['plan'] == contract
    assert result['deployed'] and not result['signed_warmup']


def test_deployment_context_cannot_exceed_any_declared_stage_or_node_ceiling(tmp_path):
    contract = plan()
    contract['stages'][0]['context_limit'] = 1024
    entries = nodes()
    entries['stage-1']['max_context'] = 512
    deployment = RingDeployment(contract, entries, state_dir=tmp_path)
    assert deployment.max_context == 512
    for stage in contract['stages']:
        argv, _ = deployment.stage_command(stage, entries[stage['node_id']])
        assert argv[argv.index('--max-ctx') + 1] == '512'
    contract['execution'] = {'max_context': 256}
    assert RingDeployment(contract, entries, state_dir=tmp_path).max_context == 256


@pytest.mark.parametrize('cap', [0, -1, True, '8192'])
def test_invalid_node_context_is_rejected_before_remote_launch(tmp_path, cap):
    entries = nodes()
    entries['stage-2']['max_context'] = cap
    with pytest.raises(ValueError, match='max_context'):
        RingDeployment(plan(), entries, state_dir=tmp_path)
