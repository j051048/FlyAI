"""Commit grouping starts only after target verification, without delaying tokens."""
import pytest

from test_v4_pipe import _ScriptedDsparkRing, VP
from test_v4_gateway import Pipe, backend, job, GW


@pytest.mark.parametrize("maximum,eos,expected", [
    (6, (), [10, 21, 31, 32, 33, 44]),
    (10, (32,), [10, 21, 31, 32]),
    (4, (), [10, 21, 31, 32]),
])
def test_serial_verified_groups_obey_eos_and_maximum_on_real_sockets(maximum, eos, expected):
    ring = _ScriptedDsparkRing(10, [([21], [31, 32, 33]), ([31, 32, 33, 44], [51, 52, 53])])
    visible, groups = [], []
    def emit(token): visible.append(token)
    def many(tokens):
        groups.append(list(tokens)); visible.extend(tokens)
    emit.commit_many = many
    try:
        result = VP.coordinate_dspark(ring.pipe_a, ring.ret_b, [1, 2, 3], maximum,
            eos_ids=eos, timeout=3, on_token=emit)
        assert result["tokens"] == visible == expected
        assert groups == [[21], expected[2:]]
    finally:
        ring.close()


def test_full_width_plus_bonus_respects_journal_transaction_bound():
    drafts = list(range(31, 95))
    ring = _ScriptedDsparkRing(10, [([21], drafts), (drafts+[95], drafts)])
    visible, groups = [], []
    def emit(token): visible.append(token)
    def many(tokens):
        assert len(tokens) <= 64
        groups.append(list(tokens)); visible.extend(tokens)
    emit.commit_many = many
    try:
        result = VP.coordinate_dspark(ring.pipe_a, ring.ret_b, [1, 2, 3], 67, timeout=3, on_token=emit)
        assert result["tokens"] == visible == [10, 21]+list(range(31, 96))
        assert list(map(len, groups)) == [1, 64, 1]
    finally:
        ring.close()


class GroupedPipe(Pipe):
    """Synthetic verified producer; exercises actual gateway replay and Job APIs."""
    def coordinate(self, *args, cancel_check, expected_by_signer, strict_job_binding=False, **kwargs):
        downstream = kwargs["on_token"]
        assert callable(downstream.commit_many)
        pending = []
        def collect(token):
            pending.append(token)
            if self.fail_first and self.calls == 1 and len(pending) == 2:
                downstream.commit_many(pending)
                pending.clear()
        kwargs["on_token"] = collect
        result = super().coordinate(*args, cancel_check=cancel_check,
            expected_by_signer=expected_by_signer, strict_job_binding=strict_job_binding, **kwargs)
        downstream.commit_many(pending)
        return result
    coordinate_dspark = coordinate
    coordinate_dspark_pipelined = coordinate


@pytest.mark.parametrize("mismatch", [False, True])
def test_grouped_replay_validates_every_replayed_token_before_publishing_fresh_suffix(mismatch):
    producer = GroupedPipe(fail_first=True, mismatch=mismatch)
    service, request = backend(producer), job(maximum=6)
    try:
        if mismatch:
            with pytest.raises(GW.ReplayMismatch):
                service.execute(request, request.emitter(), request.check_stop)
            assert request.tokens == [65, 66]
        else:
            result = service.execute(request, request.emitter(), request.check_stop)
            assert result["proof"]["verified"]
            assert request.tokens == list(range(65, 71))
            assert result["recovery"]["replayed_committed_tokens"] == 2
    finally:
        service.close()
