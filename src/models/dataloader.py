"""
src/models/dataloader.py

PyTorch DataLoader with month one-hot encoding, safe file handling, and static grid alignment.
"""

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr


@dataclass
class EALSTMTrainData:
    historical: Union[np.ndarray, torch.Tensor]  # (N, seq_len, dynamic_dim)
    static: Union[np.ndarray, torch.Tensor]  # (N, static_dim)
    pred_months: Union[np.ndarray, torch.Tensor]  # (N,) month integers
    latlons: Union[np.ndarray, torch.Tensor]  # (N, 2)
    y: Optional[Union[np.ndarray, torch.Tensor]] = None  # (N, 1)

    def to_device(self, device: torch.device) -> "EALSTMTrainData":
        def _cvt(t):
            if t is None:
                return None
            if isinstance(t, np.ndarray):
                return torch.from_numpy(t).float().to(device)
            return t.float().to(device)

        return EALSTMTrainData(
            historical=_cvt(self.historical),
            static=_cvt(self.static),
            pred_months=_cvt(self.pred_months),
            latlons=_cvt(self.latlons),
            y=_cvt(self.y) if self.y is not None else None,
        )


class EALSTMDataLoader:
    def __init__(
        self,
        data_path: Union[str, Path] = Path("data"),
        mode: str = "train",
        experiment: str = "one_month_forecast",
        batch_size: int = 256,
        shuffle: bool = True,
        normalize: bool = True,
        monthly_aggs: bool = True,
        pred_months: Optional[List[int]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.data_path = Path(data_path)
        self.mode = mode
        self.experiment = experiment
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.monthly_aggs = monthly_aggs
        self.device = torch.device(device)

        self.features_dir = self.data_path / "features" / experiment / mode
        assert self.features_dir.exists(), (
            f"Features directory does not exist: {self.features_dir}"
        )

        self.data_folders = sorted([
            f
            for f in self.features_dir.iterdir()
            if (f / "x.nc").exists() and (f / "y.nc").exists()
        ])

        # Parse month from folder name '{year}_{month}' using index [1]
        if pred_months is not None:
            self.data_folders = [
                f
                for f in self.data_folders
                if int(f.name.split("_")[1]) in pred_months
            ]

        self.normalize = normalize
        self.dynamic_norm = {}
        self.static_norm = {}

        if normalize:
            with (
                self.data_path / f"features/{experiment}/normalizing_dict.pkl"
            ).open("rb") as f:
                self.dynamic_norm = pickle.load(f)

            static_norm_path = (
                self.data_path / "features/static/normalizing_dict.pkl"
            )
            if static_norm_path.exists():
                with static_norm_path.open("rb") as f:
                    self.static_norm = pickle.load(f)

        static_file = self.data_path / "features/static/data.nc"
        self.static_ds = (
            xr.open_dataset(static_file) if static_file.exists() else None
        )

    def __len__(self) -> int:
        return len(self.data_folders)

    def _extract_pairs(self, folder: Path) -> EALSTMTrainData:
        # Context manager ensures NetCDF descriptors close promptly
        with xr.open_dataset(folder / "x.nc") as x_ds, xr.open_dataset(folder / "y.nc") as y_ds:
            y_var = list(y_ds.data_vars)[0]
            target_month = int(folder.name.split("_")[1])

            lats = x_ds.lat.values
            lons = x_ds.lon.values
            n_lat, n_lon = len(lats), len(lons)

            # Build continuous dynamic channels
            dynamic_vars = [v for v in x_ds.data_vars if v != "spatial_ref"]
            dyn_list = []

            for v in dynamic_vars:
                da = x_ds[v].values
                if self.normalize and v in self.dynamic_norm:
                    da = (
                        da - self.dynamic_norm[v]["mean"]
                    ) / self.dynamic_norm[v]["std"]
                dyn_list.append(da)

                if self.monthly_aggs:
                    sp_mean = np.nanmean(da, axis=(1, 2), keepdims=True)
                    sp_broadcast = np.broadcast_to(sp_mean, da.shape)
                    dyn_list.append(sp_broadcast)

            dyn_arr = np.stack(dyn_list, axis=-1)

            # Month of year one-hot encoding (11 channels per Lees et al. Table 1)
            months = x_ds.time.dt.month.values
            month_oh = np.eye(12, dtype=np.float32)[months - 1][:, :11]
            month_oh_grid = np.broadcast_to(
                month_oh[:, np.newaxis, np.newaxis, :],
                (len(months), n_lat, n_lon, month_oh.shape[-1]),
            )
            dyn_arr = np.concatenate([dyn_arr, month_oh_grid], axis=-1)

            time_len, _, _, n_feats = dyn_arr.shape
            dyn_flat = dyn_arr.transpose(1, 2, 0, 3).reshape(
                n_lat * n_lon, time_len, n_feats
            )

            # Spatial meshgrid
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            latlon_flat = np.stack(
                [lat_grid.flatten(), lon_grid.flatten()], axis=-1
            )

            # Static features alignment
            static_list = []
            if self.static_ds is not None:
                static_aligned = self.static_ds.sel(lat=x_ds.lat, lon=x_ds.lon)
                for v in static_aligned.data_vars:
                    if v == "spatial_ref":
                        continue
                    s_da = static_aligned[v].values
                    if self.normalize and v in self.static_norm:
                        s_da = (
                            s_da - self.static_norm[v]["mean"]
                        ) / self.static_norm[v]["std"]
                    static_list.append(s_da)
                static_flat = np.stack(static_list, axis=-1).reshape(
                    n_lat * n_lon, -1
                )
            else:
                static_flat = np.zeros((n_lat * n_lon, 1))

            # Target normalization
            y_vals = y_ds[y_var].values.squeeze()
            if self.normalize and y_var in self.dynamic_norm:
                y_vals = (
                    y_vals - self.dynamic_norm[y_var]["mean"]
                ) / self.dynamic_norm[y_var]["std"]
            y_flat = y_vals.flatten()[:, np.newaxis]

            # Drop missing and non-finite (NaN / ±inf) values
            valid = np.isfinite(y_flat.squeeze()) & np.isfinite(dyn_flat).all(
                axis=(1, 2)
            )
            if static_flat.shape[-1] > 0:
                valid = valid & np.isfinite(static_flat).all(axis=1)

            pred_months_arr = np.full(
                (n_lat * n_lon,), target_month, dtype=np.int64
            )

            return EALSTMTrainData(
                historical=dyn_flat[valid],
                static=static_flat[valid],
                pred_months=pred_months_arr[valid],
                latlons=latlon_flat[valid],
                y=y_flat[valid],
            )

    def get_batches(
        self,
    ) -> Iterator[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        folders = list(self.data_folders)
        if self.shuffle:
            np.random.shuffle(folders)

        for folder in folders:
            data = self._extract_pairs(folder).to_device(self.device)
            n_samples = data.historical.shape[0]
            if n_samples == 0:
                continue

            indices = np.arange(n_samples)
            if self.shuffle:
                np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                yield (
                    data.historical[batch_idx],
                    data.static[batch_idx],
                    data.pred_months[batch_idx],
                    data.y[batch_idx],
                )