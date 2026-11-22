"""
src/analysis/evaluation.py

Full evaluation suite: global metrics, confusion matrix, county timeseries, and SHAP ranking.
"""

import json
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import torch
import xarray as xr

from ..models.dataloader import EALSTMDataLoader
from ..models.ealstm import EALSTMNetwork

# Approximate bounding boxes for Kenya's primary ASAL drought counties
COUNTY_BOUNDS = {
    "Marsabit": {"lat_min": 1.5, "lat_max": 4.5, "lon_min": 36.5, "lon_max": 39.2},
    "Mandera": {"lat_min": 2.5, "lat_max": 4.2, "lon_min": 39.8, "lon_max": 41.9},
    "Turkana": {"lat_min": 1.0, "lat_max": 4.8, "lon_min": 34.8, "lon_max": 36.8},
    "Wajir": {"lat_min": 0.5, "lat_max": 3.5, "lon_min": 38.8, "lon_max": 41.0},
}


def classify_vci(values: np.ndarray) -> np.ndarray:
    classes = np.zeros_like(values, dtype=int)
    classes[values < 10] = 1
    classes[(values >= 10) & (values < 20)] = 2
    classes[(values >= 20) & (values < 35)] = 3
    classes[(values >= 35) & (values < 50)] = 4
    classes[values >= 50] = 5
    return classes


class ModelEvaluator:
    def __init__(
        self,
        data_path: Union[str, Path] = Path("data"),
        target_variable: str = "modis_VCI3M",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.data_path = Path(data_path)
        self.target_variable = target_variable
        self.device = torch.device(device)
        self.analysis_dir = self.data_path / "models/one_month_forecast/ealstm/analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        with (self.data_path / "features/one_month_forecast/normalizing_dict.pkl").open("rb") as f:
            self.norm_dict = pickle.load(f)

    def denormalize(self, array: np.ndarray) -> np.ndarray:
        mean = self.norm_dict[self.target_variable]["mean"]
        std = self.norm_dict[self.target_variable]["std"]
        return (array * std) + mean

    def evaluate_test_set(self, model: EALSTMNetwork) -> Dict[str, float]:
        test_loader = EALSTMDataLoader(
            data_path=self.data_path,
            mode="test",
            batch_size=512,
            shuffle=False,
            device=str(self.device),
        )
        model.eval()

        all_preds, all_true, all_static, all_latlons = [], [], [], []

        with torch.no_grad():
            for x_d, x_s, _, y in test_loader.get_batches():
                preds = model(x_d, x_s)
                all_preds.append(preds.cpu().numpy())
                all_true.append(y.cpu().numpy())
                all_static.append(x_s.cpu().numpy())

        # Collect spatial coords from test folders
        for folder in test_loader.data_folders:
            pair = test_loader._extract_pairs(folder)
            all_latlons.append(pair.latlons)

        preds_raw = self.denormalize(np.concatenate(all_preds).flatten())
        true_raw = self.denormalize(np.concatenate(all_true).flatten())
        latlons = np.concatenate(all_latlons, axis=0)

        rmse = float(np.sqrt(mean_squared_error(true_raw, preds_raw)))
        r2 = float(r2_score(true_raw, preds_raw))
        results = {"overall_rmse": rmse, "overall_r2": r2}
        print(f"Test Set Evaluation -> Overall RMSE: {rmse:.3f} | R2: {r2:.3f}")

        with (self.analysis_dir / "test_results.json").open("w") as f:
            json.dump(results, f, indent=4)

        self.plot_confusion_matrix(true_raw, preds_raw)
        self.plot_scatter(true_raw, preds_raw)
        self.cluster_static_embeddings(model, np.concatenate(all_static))
        self.evaluate_counties(model, test_loader)

        return results

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        true_cls = classify_vci(y_true)
        pred_cls = classify_vci(y_pred)
        cm = confusion_matrix(true_cls, pred_cls, labels=[1, 2, 17, 23, 25], normalize="true")

        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.matshow(cm, cmap="Blues")
        fig.colorbar(cax)

        for i in range(5):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if cm[i, j] < 0.5 else "white",
                )

        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([f"Class {i}" for i in range(1, 6)])
        ax.set_yticklabels([f"Class {i}" for i in range(1, 6)])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Normalized Confusion Matrix (VCI Deficit Classes)", pad=20)
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "confusion_matrix.png", dpi=200)
        plt.close()

    def plot_scatter(self, y_true: np.ndarray, y_pred: np.ndarray):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.05, s=2, c="#d95f02")
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", label="1:1 line")
        ax.set_xlabel("Observed VCI")
        ax.set_ylabel("Predicted VCI")
        ax.set_title("EA-LSTM Predictions vs Observations")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "scatter_obs_pred.png", dpi=200)
        plt.close()

    def cluster_static_embeddings(self, model: EALSTMNetwork, static_data: np.ndarray, k: int = 4):
        model.eval()
        with torch.no_grad():
            x_s_tensor = torch.from_numpy(static_data).float().to(self.device)
            embeddings = model.get_static_embeddings(x_s_tensor).cpu().numpy()

        kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        np.save(self.analysis_dir / "cluster_labels.npy", kmeans.labels_)
        np.save(self.analysis_dir / "static_embeddings.npy", embeddings)

    def evaluate_counties(self, model: EALSTMNetwork, test_loader: EALSTMDataLoader):
        """
        Extracts monthly observed vs. predicted timeseries and computes R2
        for Marsabit, Mandera, Turkana, and Wajir (Table 4 & Figure 9).
        """
        model.eval()
        county_ts = {c: {"dates": [], "obs": [], "pred": []} for c in COUNTY_BOUNDS}

        for folder in test_loader.data_folders:
            date_str = folder.name
            pair = test_loader._extract_pairs(folder).to_device(self.device)

            with torch.no_grad():
                preds = model(pair.historical, pair.static).cpu().numpy().flatten()
                obs = pair.y.cpu().numpy().flatten()

            preds_raw = self.denormalize(preds)
            obs_raw = self.denormalize(obs)
            lats = pair.latlons[:, 0]
            lons = pair.latlons[:, 1]

            for county, bbox in COUNTY_BOUNDS.items():
                mask = (
                    (lats >= bbox["lat_min"]) & (lats <= bbox["lat_max"]) &
                    (lons >= bbox["lon_min"]) & (lons <= bbox["lon_max"])
                )
                if np.sum(mask) > 0:
                    county_ts[county]["dates"].append(date_str)
                    county_ts[county]["obs"].append(float(np.mean(obs_raw[mask])))
                    county_ts[county]["pred"].append(float(np.mean(preds_raw[mask])))

        # Plot 4-panel timeseries and compute R2
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        county_r2 = {}

        for idx, (county, data) in enumerate(county_ts.items()):
            ax = axes[idx // 2, idx % 2]
            obs_arr = np.array(data["obs"])
            pred_arr = np.array(data["pred"])
            r2_val = r2_score(obs_arr, pred_arr) if len(obs_arr) > 1 else float("nan")
            county_r2[county] = float(r2_val)

            ax.plot(data["dates"], obs_arr, "k--", label="Observed VCI", linewidth=1.5)
            ax.plot(data["dates"], pred_arr, color="#e66101", label="EA-LSTM", linewidth=1.5)
            ax.set_title(f"{county.upper()} — R²: {r2_val:.2f}")
            ax.set_ylabel("VCI3M")
            ax.tick_params(axis="x", rotation=45)
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.analysis_dir / "county_timeseries.png", dpi=200)
        plt.close()

        with (self.analysis_dir / "county_r2_scores.json").open("w") as f:
            json.dump(county_r2, f, indent=4)
        print("County-level evaluation completed. Summary:", county_r2)

    def compute_shap_ranking(
        self,
        model: EALSTMNetwork,
        feature_names: List[str],
        num_samples: int = 100,
    ):
        """
        Computes feature importance rankings using gradient-based attribution
        as in Lees et al. Figure 11.
        """
        import shap

        test_loader = EALSTMDataLoader(
            data_path=self.data_path,
            mode="test",
            batch_size=num_samples,
            shuffle=True,
            device=str(self.device),
        )
        x_d, x_s, _, _ = next(test_loader.get_batches())

        # Define wrapper mapping concatenated (x_d, x_s) to predictions
        d_len = x_d.shape[1] * x_d.shape[2]
        dyn_shape = x_d.shape[1:]

        class ModelWrapper(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, flat_inputs):
                xd = flat_inputs[:, :d_len].reshape(-1, *dyn_shape)
                xs = flat_inputs[:, d_len:]
                return self.net(xd, xs)

        wrapper = ModelWrapper(model).to(self.device)
        flat_x = torch.cat([x_d.reshape(x_d.shape, -1), x_s], dim=1)

        explainer = shap.GradientExplainer(wrapper, flat_x[:50])
        shap_values = explainer.shap_values(flat_x[50:100])

        if isinstance(shap_values, list):
            shap_values = shap_values

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        np.save(self.analysis_dir / "mean_abs_shap.npy", mean_abs_shap)

        # Plot feature importance ranking
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(mean_abs_shap)[::-1][:len(feature_names)]
        plt.bar(range(len(sorted_idx)), mean_abs_shap[sorted_idx], color="#1f78b4")
        plt.xticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx], rotation=90)
        plt.ylabel("Mean Absolute Shapley Value")
        plt.title("EA-LSTM Feature Importance Ranking")
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "shap_importance_ranking.png", dpi=200)
        plt.close()
        print("SHAP feature ranking generated successfully.")