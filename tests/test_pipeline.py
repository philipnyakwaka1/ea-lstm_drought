"""
tests/test_pipeline.py

Unit tests verifying tensor shapes, month one-hot dimensions, gradient flow, and checkpointing.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.ealstm import EALSTMNetwork


def test_ea_lstm_pipeline():
    batch_size = 16
    seq_len = 3

    # 6 continuous dynamic vars * 2 (with monthly spatial means) + 11 month one-hot channels = 23
    n_dyn = 23

    # 7 static vars + 7 dynamic pixel climatologies = 14
    n_stat = 14
    hidden_size = 64

    print("Running Pre-Flight Test: Instantiating Tensors...")
    x_dyn = torch.randn(batch_size, seq_len, n_dyn)
    x_stat = torch.randn(batch_size, n_stat)
    y_target = torch.randn(batch_size, 1)

    model = EALSTMNetwork(
        dynamic_size=n_dyn,
        static_size=n_stat,
        hidden_size=hidden_size,
    )

    # 1. Forward Pass
    pred = model(x_dyn, x_stat)
    assert pred.shape == (batch_size, 1), f"Incorrect prediction shape: {pred.shape}"

    # 2. Unweighted Backward Pass
    loss = F.smooth_l1_loss(pred, y_target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient is None for parameter: {name}"

    # 3. Static Gate Activations
    gate_act = model.get_static_embeddings(x_stat)
    assert gate_act.shape == (batch_size, hidden_size)
    assert (gate_act >= 0.0).all() and (gate_act <= 1.0).all(), "Gate values outside [1]"

    # 4. Checkpoint Serialization Test
    test_ckpt = Path("test_checkpoint.pt")
    torch.save(model.state_dict(), test_ckpt)
    model.load_state_dict(torch.load(test_ckpt))
    test_ckpt.unlink()

    print("--> All pre-flight tests PASSED successfully.")


if __name__ == "__main__":
    test_ea_lstm_pipeline()