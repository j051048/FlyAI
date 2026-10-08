import pytest
from shard.performance import summarize_runs, validate_measurement


def result():
    return {'tokens':[1,2,3,4], 'proof':{'verified':True}, 'metrics':{
        'schema':'shard-pipeline-metrics/2','committed_tokens':4,'new_tokens':2,'new_decode_tokens':1,
        'decode_s':.1,'request_s':.3,'ttft_s':.2}}


def test_resume_and_prefill_are_excluded_from_decode_throughput():
    measured=validate_measurement(result())
    assert measured['decode_tok_s']==10
    assert measured['request_tok_s']==pytest.approx(2/.3)


def test_workloads_are_never_combined_into_one_throughput():
    records=[{'cohort_id':'c','configuration_sha256':'d','workload':w,'result':result()} for w in ('novel','copy')]
    summary=summarize_runs(records)
    assert len(summary['groups'])==2
    assert {r['workload'] for r in summary['groups']}=={'novel','copy'}


def test_invalid_token_count_and_unverified_output_cannot_be_a_benchmark():
    value=result();value['metrics']['committed_tokens']=5
    with pytest.raises(ValueError): validate_measurement(value)
    value=result();value['proof']['verified']=False
    with pytest.raises(ValueError): summarize_runs([{'result':value}])
