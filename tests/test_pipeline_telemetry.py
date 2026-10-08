import json
import time
from shard.pipeline_telemetry import StageTelemetry


def test_local_durations_are_bounded_and_do_not_call_gpu_synchronize(tmp_path):
    path=tmp_path/'stage.json'
    telemetry=StageTelemetry(node_id='a',cohort_id='c',index=0,path=path,window=3)
    for value in (1,2,3,4,5): telemetry.record('send_queue_wait_ms',value)
    with telemetry.measure('decode','cpu'): time.sleep(.001)
    telemetry.write()
    body=json.loads(path.read_text())
    assert body['durations']['send_queue_wait_ms']['samples']==3
    assert body['durations']['send_queue_wait_ms']['p50']==4
    assert body['durations']['decode_host_ms']['p50']>0
    assert 'decode_gpu_ms' not in body['durations']
    assert body['network_probe'] is None
