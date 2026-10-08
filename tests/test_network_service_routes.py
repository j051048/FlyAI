import copy
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from shard.manifest import pub_b64
from shard.network_service import pipeline_assignment
from shard.offers import ModelCohort


def fixture():
    cohort=ModelCohort('model/test','a'*64,'checkpoint','b'*64,'fp4','runtime/1','wire/1','greedy',2)
    plan={'ring_id':'ring','stages':[
        {'id':'a','index':0,'lo':0,'hi':1,'head':True,'tail':False},
        {'id':'b','index':1,'lo':1,'hi':2,'head':False,'tail':True}],
        'planning':{'coordinator_id':'c','routes':[
            {'src':'a','dst':'b','route_id':'ab','dialer_id':'a','dial_endpoint':'127.0.0.1:30002'},
            {'src':'c','dst':'a','route_id':'entry','dialer_id':'c','dial_endpoint':'127.0.0.1:30001'},
            {'src':'b','dst':'c','route_id':'return','dialer_id':'c','dial_endpoint':'127.0.0.1:30003'}]}}
    offers=[{'gpu_uuid':'gpu-'+s,'public_key':pub_b64(Ed25519PrivateKey.generate())} for s in ('a','b')]
    formation={'head':'127.0.0.1:30001','tail':'127.0.0.1:30003',
               'route_endpoints':{'ab':'127.0.0.1:30002','entry':'127.0.0.1:30001','return':'127.0.0.1:30003'}}
    return plan,cohort,offers,formation


def test_selected_route_controls_actual_engine_next_and_return_dialer():
    plan,cohort,offers,formation=fixture()
    value=pipeline_assignment(plan,cohort,offers,formation)
    assert value['stages'][0]['next_endpoint']=='127.0.0.1:30002'


@pytest.mark.parametrize('kind',['port','dialer','missing','override'])
def test_cost_measurement_cannot_be_reused_for_different_actual_route(kind):
    plan,cohort,offers,formation=fixture()
    if kind=='port': formation['route_endpoints']['ab']='127.0.0.1:31000'
    if kind=='dialer': plan['planning']['routes'][-1]['dialer_id']='b'
    if kind=='missing': del plan['planning']['routes'][0]['dial_endpoint']
    if kind=='override': formation['stage_next']={'a':'127.0.0.1:31000'}
    with pytest.raises(ValueError): pipeline_assignment(plan,cohort,offers,formation)
