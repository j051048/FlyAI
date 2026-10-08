import torch
import specpipe


def test_float32_differences_lost_by_fp16_are_still_committed():
    a=torch.tensor([1.00001],dtype=torch.float32)
    b=torch.tensor([1.00002],dtype=torch.float32)
    assert torch.equal(a.to(torch.float16),b.to(torch.float16))
    assert specpipe._act_digest(a)!=specpipe._act_digest(b)


def test_shape_and_dtype_are_bound_and_contiguous_layout_is_equivalent():
    a=torch.arange(6,dtype=torch.bfloat16).reshape(2,3)
    assert specpipe._act_digest(a)!=specpipe._act_digest(a.reshape(3,2))
    assert specpipe._act_digest(a)!=specpipe._act_digest(a.to(torch.float16))
    assert specpipe._act_digest(a.t())==specpipe._act_digest(a.t().contiguous())
