"""A discriminating random selftest, without weakening token/bit acceptance."""
import math

import pytest

torch = pytest.importorskip("torch")
R = pytest.importorskip("v4_ref_cpu")

PROMPT = (168, 15, 493, 72, 22)
NEW = 6


def continuation(model):
    ids = torch.tensor([PROMPT])
    token, logits, _ = model(ids)
    tokens, observed, margins = [], [], []
    for i in range(NEW):
        tokens.append(int(token.item()))
        observed.append(logits.clone())
        top = logits.topk(2).values[0]
        margins.append(float(top[0] - top[1]))
        if i + 1 < NEW:
            token, logits, _ = model(token.reshape(1, 1), len(PROMPT) + i)
    return tokens, observed, margins


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_fixed_seed_fixture_is_discriminating_and_bit_reproducible(threads):
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        args = R.cpu_args()
        first = R.init_fingerprint_fixture(R.build_oracle(args, seed=0), args)
        second = R.init_fingerprint_fixture(R.build_oracle(args, seed=0), args)
        a, a_logits, a_margins = continuation(first)
        b, b_logits, b_margins = continuation(second)
        assert len(set(a)) == NEW  # Stronger than the existing selftest fingerprint minimum.
        assert a == b
        assert all(torch.equal(x, y) for x, y in zip(a_logits, b_logits))
        assert a_margins == b_margins
        assert min(a_margins) > 1e-2  # This fixture is not a near-tie lottery.
    finally:
        torch.set_num_threads(old)


def test_initializer_changes_only_main_residual_output_weights_and_consumes_no_rng():
    args = R.cpu_args()
    model = R.build_oracle(args, seed=0)
    parameters = {name: value.clone() for name, value in model.named_parameters()}
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    outputs = {id(layer.attn.wo_b.weight) for layer in model.layers}
    outputs.update(id(expert.w2.weight) for layer in model.layers
                   for expert in list(layer.ffn.experts) + [layer.ffn.shared_experts])
    rng = torch.get_rng_state().clone()
    assert R.init_fingerprint_fixture(model, args) is model
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in model.named_parameters():
        expected = parameters[name] / math.sqrt(2 * args.n_layers) if id(value) in outputs else parameters[name]
        assert torch.equal(value, expected), name
    for name, value in model.named_buffers():
        assert torch.equal(value, buffers[name]), name
    assert model._fingerprint_fixture_init["divisor"] == 4


def test_ordinary_oracle_remains_the_original_seeded_initialization():
    model = R.build_oracle(seed=0)
    assert not hasattr(model, "_fingerprint_fixture_init")
    second = R.build_oracle(seed=0)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), second.parameters()))


def test_double_initialization_is_rejected_before_weights_change():
    args = R.cpu_args()
    model = R.init_fingerprint_fixture(R.build_oracle(args), args)
    weights = {name: value.clone() for name, value in model.named_parameters()}
    with pytest.raises(ValueError, match="already initialized"):
        R.init_fingerprint_fixture(model, args)
    assert all(torch.equal(value, weights[name]) for name, value in model.named_parameters())


def test_quantized_or_mismatched_fixture_is_rejected_before_mutation():
    args = R.cpu_args()
    model = R.build_oracle(args)
    before = model.layers[0].attn.wo_b.weight.clone()
    args.expert_dtype = "fp4"
    with pytest.raises(ValueError, match="unquantized"):
        R.init_fingerprint_fixture(model, args)
    assert torch.equal(before, model.layers[0].attn.wo_b.weight)
    args.expert_dtype = None
    args.n_layers += 1
    with pytest.raises(ValueError, match="matching"):
        R.init_fingerprint_fixture(model, args)
    assert torch.equal(before, model.layers[0].attn.wo_b.weight)
