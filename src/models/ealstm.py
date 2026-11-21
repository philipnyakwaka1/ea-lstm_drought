"""
src/models/ealstm.py

PyTorch implementation of the Entity-Aware LSTM with unweighted Smooth L1 loss.
"""

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataloader import EALSTMDataLoader


class OrgEALSTMCell(nn.Module):
    def __init__(
        self,
        input_size_dyn: int,
        input_size_stat: int,
        hidden_size: int,
        batch_first: bool = True,
        initial_forget_bias: float = 0.0,
    ):
        super().__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size_dyn, 3 * hidden_size)
        )
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size)
        )
        self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))

        self.weight_sh = nn.Parameter(
            torch.FloatTensor(input_size_stat, hidden_size)
        )
        self.bias_s = nn.Parameter(torch.FloatTensor(hidden_size))

        self.reset_parameters(initial_forget_bias)

    def reset_parameters(self, initial_forget_bias: float):
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_sh.data)

        weight_hh_data = torch.eye(self.hidden_size).repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, 0.0)
        nn.init.constant_(self.bias_s.data, 0.0)

        if initial_forget_bias != 0:
            self.bias.data[: self.hidden_size] = initial_forget_bias

    def forward(
        self, x_d: torch.Tensor, x_s: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.batch_first:
            x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_t = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_t = x_d.data.new(batch_size, self.hidden_size).zero_()

        bias_s_batch = self.bias_s.unsqueeze(0).expand(
            batch_size, *self.bias_s.size()
        )
        i = torch.sigmoid(torch.addmm(bias_s_batch, x_s, self.weight_sh))

        bias_batch = self.bias.unsqueeze(0).expand(
            batch_size, *self.bias.size()
        )

        h_n = []
        for t in range(seq_len):
            gates = torch.addmm(bias_batch, h_t, self.weight_hh) + torch.mm(
                x_d[t], self.weight_ih
            )
            f, o, g = gates.chunk(3, dim=1)

            c_t = torch.sigmoid(f) * c_t + i * torch.tanh(g)
            h_t = torch.sigmoid(o) * torch.tanh(c_t)
            h_n.append(h_t)

        h_n = torch.stack(h_n, dim=0)
        if self.batch_first:
            h_n = h_n.transpose(0, 1)

        return h_n, (h_t, c_t)


class EALSTMNetwork(nn.Module):
    def __init__(
        self,
        dynamic_size: int,
        static_size: int,
        hidden_size: int = 64,
        rnn_dropout: float = 0.25,
        dense_features: Optional[List[int]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.cell = OrgEALSTMCell(
            input_size_dyn=dynamic_size,
            input_size_stat=static_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(rnn_dropout)

        layers = []
        in_dim = hidden_size
        dense_dims = dense_features if dense_features is not None else []
        for d in dense_dims:
            layers.append(nn.Linear(in_dim, d))
            layers.append(nn.ReLU())
            in_dim = d

        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        out, _ = self.cell(x_d, x_s)
        last_hidden = self.dropout(out[:, -1, :])
        return self.head(last_hidden)

    def get_static_embeddings(self, x_s: torch.Tensor) -> torch.Tensor:
        batch_size = x_s.shape[0]
        bias_s_batch = self.cell.bias_s.unsqueeze(0).expand(
            batch_size, *self.cell.bias_s.size()
        )
        return torch.sigmoid(
            torch.addmm(bias_s_batch, x_s, self.cell.weight_sh)
        )


class EARecurrentModel:
    def __init__(
        self,
        data_path: Union[str, Path] = Path("data"),
        hidden_size: int = 64,
        rnn_dropout: float = 0.25,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.data_path = Path(data_path)
        self.hidden_size = hidden_size
        self.rnn_dropout = rnn_dropout
        self.learning_rate = learning_rate
        self.device = torch.device(device)

        self.model_dir = (
            self.data_path / "models" / "one_month_forecast" / "ealstm"
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[EALSTMNetwork] = None

    def _init_model(self, sample_dyn: torch.Tensor, sample_stat: torch.Tensor):
        dyn_dim = sample_dyn.shape[-1]
        stat_dim = sample_stat.shape[-1]
        self.model = EALSTMNetwork(
            dynamic_size=dyn_dim,
            static_size=stat_dim,
            hidden_size=self.hidden_size,
            rnn_dropout=self.rnn_dropout,
        ).to(self.device)

    def train(
        self,
        num_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
    ):
        train_loader = EALSTMDataLoader(
            data_path=self.data_path,
            mode="train",
            batch_size=batch_size,
            shuffle=True,
            device=str(self.device),
        )

        val_loader = EALSTMDataLoader(
            data_path=self.data_path,
            mode="val",
            batch_size=batch_size,
            shuffle=False,
            device=str(self.device),
        )

        first_batch = next(train_loader.get_batches())
        if self.model is None:
            self._init_model(first_batch[0], first_batch[1])

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        # Unweighted Smooth L1 loss per Lees et al. (2022)
        criterion = nn.SmoothL1Loss()

        best_val_rmse = float("inf")
        epochs_no_improve = 0

        print(
            f"Training EA-LSTM on {self.device} (Max Epochs: {num_epochs}, Patience: {patience})..."
        )

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            train_losses = []
            for x_d, x_s, _, y in train_loader.get_batches():
                optimizer.zero_grad()
                pred = self.model(x_d, x_s)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for x_d, x_s, _, y in val_loader.get_batches():
                    pred = self.model(x_d, x_s)
                    v_loss = F.mse_loss(pred, y)
                    val_losses.append(v_loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_rmse = (
                math.sqrt(np.mean(val_losses)) if val_losses else float("nan")
            )

            print(
                f"Epoch {epoch:03d} | Train Smooth L1: {avg_train_loss:.4f} | Val RMSE: {avg_val_rmse:.4f}"
            )

            if avg_val_rmse < best_val_rmse:
                best_val_rmse = avg_val_rmse
                epochs_no_improve = 0
                torch.save(
                    self.model.state_dict(), self.model_dir / "best_model.pt"
                )
                print(
                    f"--> Saved best model weights (Val RMSE: {best_val_rmse:.4f})"
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        self.model.load_state_dict(
            torch.load(self.model_dir / "best_model.pt")
        )