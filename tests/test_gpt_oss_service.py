"""Shared production service with synthetic signed GPT-OSS coordinator results."""
import socket
import time
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from engines.gpt_oss.network_service import GPTOSSRingBackend
from shard.manifest import pub_b64
from shard.offers import ModelCohort
from shard.pipeline_plan import build_plan
from shard.receipt import ReceiptError, ReceiptSigner
from shard.service_queue import Job


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs): return [1, 2, 3]
    def decode(self, ids, **kwargs): return "".join(chr(65 + i % 26) for i in ids)


class TransportError(RuntimeError): pass


class Runtime:
    TransportError = TransportError
    NgramDrafter = lambda self, **kw: object()
    def __init__(self, keys, *, fail_once=False, stale=False):
        self.keys, self.fail_once, self.stale = keys, fail_once, stale
        self.calls, self.peers = 0, []
    def connect_ring(self, head, tail, **kwargs):
        channels = []
        for _ in range(2):
            client, peer = socket.socketpair()
            channels.append(client); self.peers.append(peer)
        return tuple(channels)
    def coordinate_pipe(self, *args, **kwargs):
        self.calls += 1
        maximum = args[5]
        tokens = [10, 11, 12][:maximum]
        kwargs['on_commit']({'out': tokens[:1]})
        if self.fail_once and self.calls == 1: raise TransportError('lost edge')
        kwargs['on_commit']({'out': tokens})
        receipts = []
        for i, key in enumerate(self.keys):
            signer = ReceiptSigner(key, kwargs['swarm_id'], kwargs['job_id'], i, i + 1,
                'stale' if self.stale else kwargs['nonce'])
            signer.observe(str(i).encode(), str(i + 1).encode())
            receipts.append(signer.finalize())
        return {'ok': True, 'output_ids': tokens, 'receipts': receipts,
                'metrics': {'new_decode_tokens': len(tokens)-1, 'decode_s': .1}}
    def close(self):
        for peer in self.peers: peer.close()


@pytest.fixture
def backend():
    keys = [Ed25519PrivateKey.generate() for _ in range(3)]
    cohort = ModelCohort('openai/gpt-oss-120b', 'a'*64, 'fixed-checkpoint', 'b'*64,
                        'mxfp4', 'gpt-oss/1', 'pipeline/2', 'greedy-parity', 3)
    plan = build_plan({'num_hidden_layers': 3}, ring_id='oss-test', cohort_id=cohort.cohort_id,
        endpoints=['localhost:30001', 'localhost:30002', 'localhost:30003'], model_cohort=cohort.to_dict())
    for stage, key in zip(plan['stages'], keys): stage['signer_pubkey'] = pub_b64(key)
    plan['coordinator']['signer_pubkey'] = pub_b64(keys[0])
    runtime = Runtime(keys)
    result = GPTOSSRingBackend('.', plan, tokenizer=Tokenizer(), runtime=runtime, max_context=1024, coordinator_key=keys[0])
    yield result
    result.close(); runtime.close()


def make_job(backend, maximum=2):
    payload, count, maximum, timeout = backend.prepare({'messages':[{'role':'user','content':'hello'}], 'max_tokens':maximum})
    now = time.monotonic()
    return Job('test', payload, count, maximum, now + timeout, 'job', None, now, state='running')


def test_signed_warmup_and_request_on_same_backend(backend):
    assert not backend.ready()[0]
    assert backend.warmup()['proof_verified']
    assert backend.ready()[0]
    job = make_job(backend)
    result = backend.execute(job, job.commit, job.check_stop)
    assert result['tokens'] == job.tokens == [10, 11]
    assert result['proof']['verified']
    assert result['proof']['attempt_id'] == job.id + '/attempt-1'


def test_replay_emits_only_new_suffix_and_complete_final_receipts(backend):
    backend.runtime.fail_once = True
    job = make_job(backend)
    result = backend.execute(job, job.commit, job.check_stop)
    assert job.tokens == [10, 11]
    assert result['recovery']['attempts'] == 2
    assert all(r['job_id'] == job.id + '/attempt-2' for r in result['proof']['receipts'])


def test_stale_validly_signed_receipt_is_not_retried(backend):
    backend.runtime.stale = True
    job = make_job(backend)
    with pytest.raises(ReceiptError): backend.execute(job, job.commit, job.check_stop)
    assert backend.runtime.calls == 1
    assert not backend.ready()[0]


def test_old_attempt_cannot_close_new_attempt_channels(backend):
    job=make_job(backend)
    backend.execute(job,job.commit,job.check_stop)
    channels=backend._channels
    old,new=object(),object()
    backend._attempt_owner=new
    backend.abort(old)
    assert backend._channels is channels
    assert all(channel.fileno()>=0 for channel in channels)


@pytest.mark.parametrize('change', [{'temperature':1}, {'stream':'yes'}, {'reasoning_effort':'bad'}, {'max_tokens':True}])
def test_request_contract_rejects_unsupported_semantics(backend, change):
    with pytest.raises(ValueError):
        backend.prepare({'messages':[{'role':'user','content':'hello'}], **change})
