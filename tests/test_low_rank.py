import pytest

torch = pytest.importorskip("torch")
from houndoracle.experiments.low_rank import LowRankLinear, objective


def test_zero_initial_adapter_preserves_base_and_frozen_weights():
    base = torch.nn.Linear(4, 3)
    values = torch.randn(2, 4)
    before = base(values).detach()
    adapted = LowRankLinear(base, rank=2)
    assert torch.equal(adapted(values), before)
    adapted(values).sum().backward()
    assert base.weight.grad is None and adapted.lora_B.grad is not None
    assert torch.count_nonzero(adapted.lora_B.grad) > 0


def test_pair_loss_rewards_correct_relational_separation():
    labels = torch.tensor([0, 1])
    weak = torch.zeros(2, 3, requires_grad=True)
    strong = torch.tensor([[5.0, -2.0, 0.0], [-2.0, 5.0, 0.0]], requires_grad=True)
    a, ce, ranking = objective(weak, labels)
    b, _, better_ranking = objective(strong, labels)
    assert b < a and better_ranking < ranking
    a.backward()
    assert torch.isfinite(weak.grad).all()
    control, _, _ = objective(weak.detach(), labels, pair_weight=0)
    assert torch.equal(control, ce.detach())


def test_same_label_or_unpaired_batch_cannot_fake_contrastive_supervision():
    with pytest.raises(ValueError):
        objective(torch.zeros(2, 3), torch.tensor([1, 1]))
    with pytest.raises(ValueError):
        objective(torch.zeros(3, 3), torch.tensor([0, 1, 2]))


def test_checkpoint_roundtrip_and_schema_rejection_preserve_parameters():
    import copy

    from houndoracle.experiments.low_rank import restore_trainable, trainable_state

    original = torch.nn.Linear(4, 3)
    a = LowRankLinear(copy.deepcopy(original), rank=2)
    b = LowRankLinear(copy.deepcopy(original), rank=2)
    with torch.no_grad():
        a.lora_B.normal_()
    state = {n: value.clone() for n, value in trainable_state(a).items()}
    restore_trainable(b, state)
    values = torch.randn(3, 4)
    assert torch.equal(a(values), b(values))
    before = b.lora_B.detach().clone()
    with pytest.raises(ValueError):
        restore_trainable(b, {"lora_B": state["lora_B"]})
    assert torch.equal(b.lora_B, before)
