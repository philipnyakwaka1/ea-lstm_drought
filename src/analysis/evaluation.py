"""
src/analysis/evaluation.py

Enriched evaluation suite: global & drought-class metrics, confusion matrix,
scatter diagnostics, deduplicated static-embedding spatial clustering with
optional county boundary overlays, exact polygon-based county aggregation,
and categorized SHAP interpretability.
"""

import json
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import torch
import xarray as xr

from ..models.dataloader import EALSTMDataLoader
from ..models.ealstm import EALSTMNetwork

# Fallback bounding boxes used only if no county polygon file is supplied or found
COUNTY_BOUNDS = {
    "Marsabit": {"lat_min": 1.5, "lat_max": 4.5, "lon_min": 36.5, "lon_max": 39.2},
    "Mandera":  {"lat_min": 2.5, "lat_max": 4.2, "lon_min": 39.8, "lon_max": 41.9},
    "Turkana":  {"lat_min": 1.0, "lat_max": 4.8, "lon_min": 34.8, "lon_max": 36.8},
    "Wajir":    {"lat_min": 0.5, "lat_max": 3.5, "lon_min": 38.8, "lon_max": 41.0},
}

DEFAULT_TARGET_COUNTIES = ["Marsabit", "Mandera", "Turkana", "Wajir"]

DEFAULT_BOUNDARY_FILENAMES = [
    "kenya_counties.gpkg",
    "kenya_counties.geojson",
    "kenya_counties.shp",
]

CANDIDATE_COUNTY_NAME_COLS = [
    "ADM1_EN",
    "ADM1_NAME",
    "NAME_1",
    "COUNTY",
    "COUNTY_NAM",
    "County",
    "NAME",
    "Name",
    "name",
]

# NDMA / Klisch & Atzberger (2016) VCI3M drought deficit class labels
VCI_CLASS_LABELS = [
    "C1: Extreme\n(<10)",
    "C2: Severe\n(10-20)",
    "C3: Moderate\n(20-35)",
    "C4: Normal\n(35-50)",
    "C5: Above\n(>=50)",
]

# Publication-ready short aliases matching Step 6 conventions
BASE_FEATURE_ALIASES = {
    "precip": "chirps",
    "precipitation": "chirps",
    "chirps": "chirps",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
    "volumetric_soil_water_layer_3": "swvl3",
    "volumetric_soil_water_layer_4": "swvl4",
    "swvl1": "swvl1",
    "swvl2": "swvl2",
    "swvl3": "swvl3",
    "swvl4": "swvl4",
    "potential_evaporation": "pev",
    "potential_evaporation_hourly": "pev",
    "pev": "pev",
    "total_evaporation": "e",
    "total_evaporation_hourly": "e",
    "e": "e",
    "modis_vci3m": "vci3m",
    "vci3m": "vci3m",
    "modis_vci": "vci",
    "vci": "vci",
    "modis_ndvi": "ndvi",
    "ndvi": "ndvi",
    "elevation": "elevation",
    "srtm": "elevation",
    "b1": "elevation",
    "slt_coarse_one_hot": "soil_coarse",
    "slt_medium_one_hot": "soil_med",
    "slt_medium_fine_one_hot": "soil_med_fine",
    "slt_fine_one_hot": "soil_fine",
    "slt_very_fine_one_hot": "soil_very_fine",
    "slt_organic_one_hot": "soil_organic",
    "slt_tropical_organic_one_hot": "soil_trop_organic",
}

CATEGORY_COLORS = {
    "Dynamic Local": "#1f78b4",
    "Dynamic Spatial Mean": "#66a61e",
    "Month Seasonality": "#7570b3",
    "Static Land Surface": "#d95f02",
    "Static Climatology": "#e6ab02",
}


def _to_numpy(arr: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Safely converts a CPU/GPU PyTorch tensor or array-like to a NumPy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _sort_month_folders(folders: List[Path]) -> List[Path]:
    """Sorts unpadded 'YYYY_M' folder paths chronologically."""
    return sorted(folders, key=lambda f: tuple(map(int, f.name.split("_"))))


def _points_in_polygon(lons: np.ndarray, lats: np.ndarray, geom) -> np.ndarray:
    """Vectorized point-in-polygon mask for (lon, lat) pixel centers in EPSG:4326."""
    import shapely

    if hasattr(shapely, "contains_xy"):
        return np.asarray(shapely.contains_xy(geom, lons, lats), dtype=bool)

    from shapely.geometry import Point
    from shapely.prepared import prep

    prepared = prep(geom)
    return np.array([prepared.contains(Point(x, y)) for x, y in zip(lons, lats)], dtype=bool)


def _safe_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    pearson_r = _safe_pearson_r(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r,
        "bias": bias,
    }


def _alias_feature_name(raw_name: str, is_static: bool = False) -> str:
    """Maps verbose NetCDF variable names into clean publication-ready labels."""
    name = raw_name.strip()
    lower = name.lower()

    is_sp_mean = False
    is_pixel_mean = False

    if lower.endswith("_spatial_mean"):
        is_sp_mean = True
        lower = lower[: -len("_spatial_mean")]
    elif lower.endswith("_sp_mean"):
        is_sp_mean = True
        lower = lower[: -len("_sp_mean")]
    elif lower.startswith("spatial_mean_"):
        is_sp_mean = True
        lower = lower[len("spatial_mean_") :]

    if lower.startswith("mean_"):
        is_pixel_mean = True
        lower = lower[len("mean_") :]
    elif lower.endswith("_pixel_mean"):
        is_pixel_mean = True
        lower = lower[: -len("_pixel_mean")]
    elif is_static and lower.endswith("_mean"):
        is_pixel_mean = True
        lower = lower[: -len("_mean")]

    base = BASE_FEATURE_ALIASES.get(lower, lower)

    if not is_static and not is_pixel_mean and base in {"vci3m", "vci", "ndvi"}:
        base = f"{base}_lag"

    if is_pixel_mean:
        base = f"mean_{base}"
    if is_sp_mean:
        base = f"{base}_sp_mean"
    return base


def _classify_feature_category(name: str, is_static: bool) -> str:
    lower = name.lower()
    if not is_static:
        if lower.startswith("mo_") or lower.startswith("month_"):
            return "Month Seasonality"
        if lower.endswith("_sp_mean") or "spatial_mean" in lower:
            return "Dynamic Spatial Mean"
        return "Dynamic Local"
    if lower.startswith("mean_") or lower.endswith("_mean"):
        return "Static Climatology"
    return "Static Land Surface"


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
        counties_path: Optional[Union[str, Path]] = None,
        county_name_col: Optional[str] = None,
        target_counties: Optional[List[str]] = None,
    ):
        self.data_path = Path(data_path)
        self.target_variable = target_variable
        self.device = torch.device(device)
        self.analysis_dir = self.data_path / "models/one_month_forecast/ealstm/analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        self.county_name_col = county_name_col
        self.target_counties = (
            list(target_counties) if target_counties is not None else list(DEFAULT_TARGET_COUNTIES)
        )
        self._explicit_counties_path = counties_path is not None
        self.counties_path = (
            Path(counties_path) if counties_path is not None else self._find_default_counties_file()
        )
        self.counties_gdf, self.county_geometries = self._load_county_polygons()

        with (self.data_path / "features/one_month_forecast/normalizing_dict.pkl").open("rb") as f:
            self.norm_dict = pickle.load(f)

    def _find_default_counties_file(self) -> Optional[Path]:
        boundaries_dir = self.data_path / "boundaries"
        for filename in DEFAULT_BOUNDARY_FILENAMES:
            candidate = boundaries_dir / filename
            if candidate.exists():
                return candidate
        return None

    def _resolve_county_name_column(self, columns: List[str]) -> str:
        if self.county_name_col is not None:
            if self.county_name_col in columns:
                return self.county_name_col
            raise ValueError(
                f"county_name_col='{self.county_name_col}' not found in boundary file. "
                f"Available columns: {columns}"
            )

        col_lookup = {c.lower(): c for c in columns}
        for candidate in CANDIDATE_COUNTY_NAME_COLS:
            if candidate in columns:
                return candidate
            if candidate.lower() in col_lookup:
                return col_lookup[candidate.lower()]

        raise ValueError(
            "Could not auto-detect county name column in boundary file. "
            f"Pass county_name_col explicitly. Available columns: {columns}"
        )

    def _load_county_polygons(self):
        if self.counties_path is None:
            print(
                "No county boundary file found in data/boundaries/. "
                "Falling back to approximate COUNTY_BOUNDS rectangles."
            )
            return None, {}

        if not self.counties_path.exists():
            if self._explicit_counties_path:
                raise FileNotFoundError(f"County boundary file not found: {self.counties_path}")
            return None, {}

        import geopandas as gpd

        gdf = gpd.read_file(self.counties_path)
        if gdf.crs is None:
            print(f"Warning: {self.counties_path.name} has no CRS metadata; assuming EPSG:4326.")
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        name_col = self._resolve_county_name_column(list(gdf.columns))
        gdf = gdf.copy()
        gdf["_norm_name"] = gdf[name_col].astype(str).str.strip().str.lower()
        dissolved = gdf.dissolve(by="_norm_name")

        county_geometries = {}
        missing = []
        for county in self.target_counties:
            key = county.strip().lower()
            if key in dissolved.index:
                geom = dissolved.loc[key, "geometry"]
                if not geom.is_valid:
                    geom = geom.buffer(0)
                county_geometries[county] = geom
            else:
                missing.append(county)

        if missing:
            sample_available = sorted(gdf[name_col].dropna().astype(str).unique().tolist())[:15]
            print(
                f"Warning: Target counties not found in {self.counties_path.name} "
                f"(column '{name_col}'): {missing}. Sample available names: {sample_available}"
            )

        if not county_geometries:
            raise ValueError(
                f"None of target_counties={self.target_counties} matched column '{name_col}' "
                f"in {self.counties_path}."
            )

        print(
            f"Loaded {len(county_geometries)} county polygons from {self.counties_path} "
            f"(column: '{name_col}', CRS: EPSG:4326)."
        )
        return gdf, county_geometries

    def denormalize(self, array: np.ndarray) -> np.ndarray:
        mean = self.norm_dict[self.target_variable]["mean"]
        std = self.norm_dict[self.target_variable]["std"]
        return (array * std) + mean

    def discover_feature_names(
        self,
        dyn_dim: int,
        stat_dim: int,
        test_loader: Optional[EALSTMDataLoader] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Inspects engineered NetCDFs to derive exact dynamic and static feature names
        in dataloader order, formatted with clean short aliases.
        """
        if test_loader is None:
            test_loader = EALSTMDataLoader(
                data_path=self.data_path,
                mode="test",
                batch_size=16,
                shuffle=False,
                device=str(self.device),
            )

        sorted_folders = _sort_month_folders(test_loader.data_folders)
        with xr.open_dataset(sorted_folders[0] / "x.nc") as ds_x:
            raw_dyn = [v for v in ds_x.data_vars if v != "spatial_ref"]

        dyn_aliased = [_alias_feature_name(v, is_static=False) for v in raw_dyn]
        extra_dyn = dyn_dim - len(dyn_aliased)
        if extra_dyn == 11:
            dynamic_names = dyn_aliased + [f"mo_{m}" for m in range(1, 12)]
        elif extra_dyn > 0:
            dynamic_names = dyn_aliased + [f"dyn_extra_{i + 1}" for i in range(extra_dyn)]
        else:
            dynamic_names = dyn_aliased[:dyn_dim]

        static_nc = self.data_path / "features/one_month_forecast/static/data.nc"
        if static_nc.exists():
            with xr.open_dataset(static_nc) as ds_s:
                raw_stat = [v for v in ds_s.data_vars if v != "spatial_ref"]
            stat_aliased = [_alias_feature_name(v, is_static=True) for v in raw_stat]
        else:
            stat_aliased = []

        if len(stat_aliased) == stat_dim:
            static_names = stat_aliased
        elif len(stat_aliased) < stat_dim:
            static_names = stat_aliased + [
                f"static_extra_{i + 1}" for i in range(stat_dim - len(stat_aliased))
            ]
        else:
            static_names = stat_aliased[:stat_dim]

        return dynamic_names, static_names

    def evaluate_test_set(self, model: EALSTMNetwork) -> Dict[str, float]:
        test_loader = EALSTMDataLoader(
            data_path=self.data_path,
            mode="test",
            batch_size=512,
            shuffle=False,
            device=str(self.device),
        )
        model.eval()

        all_preds, all_true = [], []

        with torch.no_grad():
            for x_d, x_s, _, y in test_loader.get_batches():
                preds = model(x_d, x_s)
                all_preds.append(preds.cpu().numpy())
                all_true.append(y.cpu().numpy())

        # Collect and deduplicate static features + spatial coordinates across test folders
        sorted_folders = _sort_month_folders(test_loader.data_folders)
        folder_latlons, folder_static = [], []
        for folder in sorted_folders:
            pair = test_loader._extract_pairs(folder)
            folder_latlons.append(_to_numpy(pair.latlons))
            folder_static.append(_to_numpy(pair.static))

        latlons_all = np.concatenate(folder_latlons, axis=0)
        static_all = np.concatenate(folder_static, axis=0)
        _, unique_idx = np.unique(np.round(latlons_all, 5), axis=0, return_index=True)
        unique_latlons = latlons_all[unique_idx]
        unique_static = static_all[unique_idx]

        preds_raw = self.denormalize(np.concatenate(all_preds).flatten())
        true_raw = self.denormalize(np.concatenate(all_true).flatten())

        metrics = _compute_regression_metrics(true_raw, preds_raw)
        true_cls = classify_vci(true_raw)
        pred_cls = classify_vci(preds_raw)
        class_acc = float(np.mean(true_cls == pred_cls))
        adj_class_acc = float(np.mean(np.abs(true_cls - pred_cls) <= 1))

        results = {
            "overall_rmse": metrics["rmse"],
            "overall_mae": metrics["mae"],
            "overall_r2": metrics["r2"],
            "overall_pearson_r": metrics["pearson_r"],
            "overall_bias": metrics["bias"],
            "drought_class_accuracy": class_acc,
            "adjacent_class_accuracy": adj_class_acc,
            "num_test_samples": int(len(true_raw)),
        }
        print(
            f"Test Set Evaluation -> RMSE: {metrics['rmse']:.3f} | "
            f"MAE: {metrics['mae']:.3f} | R2: {metrics['r2']:.3f} | "
            f"r: {metrics['pearson_r']:.3f} | Class Acc: {class_acc * 100:.1f}% "
            f"(Within +/-1 Class: {adj_class_acc * 100:.1f}%)"
        )

        with (self.analysis_dir / "test_results.json").open("w") as f:
            json.dump(results, f, indent=4)

        self.plot_confusion_matrix(true_raw, preds_raw)
        self.plot_scatter(true_raw, preds_raw, metrics=metrics)
        self.cluster_static_embeddings(model, unique_static, latlons=unique_latlons)
        self.evaluate_counties(model, test_loader)

        return results

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        true_cls = classify_vci(y_true)
        pred_cls = classify_vci(y_pred)
        cm = confusion_matrix(true_cls, pred_cls, labels=[1, 2, 3, 4, 5], normalize="true")
        overall_acc = float(np.mean(true_cls == pred_cls))

        fig, ax = plt.subplots(figsize=(8, 7))
        cax = ax.matshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
        fig.colorbar(cax, label="Row-Normalized Proportion")

        for i in range(5):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black" if cm[i, j] < 0.5 else "white",
                )

        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(VCI_CLASS_LABELS, fontsize=9)
        ax.set_yticklabels(VCI_CLASS_LABELS, fontsize=9)
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xlabel("Predicted VCI3M Deficit Class")
        ax.set_ylabel("Observed VCI3M Deficit Class")
        ax.set_title(
            f"Normalized Confusion Matrix (Exact Class Accuracy: {overall_acc * 100:.1f}%)",
            pad=14,
        )
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "confusion_matrix.png", dpi=200)
        plt.close()

    def plot_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Optional[Dict[str, float]] = None,
    ):
        if metrics is None:
            metrics = _compute_regression_metrics(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(y_true, y_pred, alpha=0.05, s=2, c="#d95f02")
        min_val = min(float(y_true.min()), float(y_pred.min()))
        max_val = max(float(y_true.max()), float(y_pred.max()))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.2, label="1:1 line")
        ax.set_xlabel("Observed VCI3M")
        ax.set_ylabel("Predicted VCI3M")
        ax.set_title("EA-LSTM Predictions vs Observations (Holdout Test Set)")
        ax.legend(loc="lower right")
        ax.text(
            0.04,
            0.95,
            f"RMSE: {metrics['rmse']:.2f}\nMAE:  {metrics['mae']:.2f}\nR²:   {metrics['r2']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "scatter_obs_pred.png", dpi=200)
        plt.close()

    def cluster_static_embeddings(
        self,
        model: EALSTMNetwork,
        static_data: np.ndarray,
        latlons: Optional[np.ndarray] = None,
        k: int = 4,
    ):
        model.eval()
        with torch.no_grad():
            x_s_tensor = torch.from_numpy(static_data).float().to(self.device)
            embeddings = model.get_static_embeddings(x_s_tensor).cpu().numpy()

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
        np.save(self.analysis_dir / "cluster_labels.npy", kmeans.labels_)
        np.save(self.analysis_dir / "static_embeddings.npy", embeddings)

        if latlons is not None:
            np.save(self.analysis_dir / "cluster_latlons.npy", latlons)
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            scatter = ax.scatter(
                latlons[:, 1],
                latlons[:, 0],
                c=kmeans.labels_,
                cmap="tab10",
                s=18,
                marker="s",
                edgecolors="none",
                zorder=2,
            )
            if self.counties_gdf is not None:
                self.counties_gdf.boundary.plot(
                    ax=ax,
                    color="black",
                    linewidth=0.55,
                    alpha=0.65,
                    zorder=3,
                )
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_title(f"EA-LSTM Static Input-Gate Clusters (k={k})")
            ax.set_aspect("equal", adjustable="box")
            handles, _ = scatter.legend_elements()
            ax.legend(
                handles,
                [f"Cluster {c}" for c in range(k)],
                title="Agro-Ecological Regime",
                loc="lower left",
            )
            plt.tight_layout()
            plt.savefig(self.analysis_dir / "static_clusters_map.png", dpi=200)
            plt.close()

        print(f"Clustered {len(embeddings)} unique pixel static embeddings into {k} regimes.")

    def evaluate_counties(self, model: EALSTMNetwork, test_loader: EALSTMDataLoader):
        """
        Extracts monthly observed vs. predicted timeseries and computes regional metrics
        using exact county polygons (or COUNTY_BOUNDS fallback if no polygon file exists).
        """
        model.eval()
        use_polygons = len(self.county_geometries) > 0

        if use_polygons:
            active_counties = list(self.county_geometries.keys())
            masking_mode = "polygon"
        else:
            active_counties = [c for c in self.target_counties if c in COUNTY_BOUNDS]
            if not active_counties:
                active_counties = list(COUNTY_BOUNDS.keys())
            masking_mode = "bbox_fallback"

        county_ts = {
            c: {"dates": [], "obs": [], "pred": [], "pixel_counts": []}
            for c in active_counties
        }
        sorted_folders = _sort_month_folders(test_loader.data_folders)

        for folder in sorted_folders:
            date_str = folder.name
            pair = test_loader._extract_pairs(folder).to_device(self.device)

            with torch.no_grad():
                preds = model(pair.historical, pair.static).cpu().numpy().flatten()
                obs = pair.y.cpu().numpy().flatten()

            preds_raw = self.denormalize(preds)
            obs_raw = self.denormalize(obs)
            latlons_np = _to_numpy(pair.latlons)
            lats = latlons_np[:, 0]
            lons = latlons_np[:, 1]

            for county in active_counties:
                if use_polygons:
                    mask = _points_in_polygon(lons, lats, self.county_geometries[county])
                else:
                    bbox = COUNTY_BOUNDS[county]
                    mask = (
                        (lats >= bbox["lat_min"]) & (lats <= bbox["lat_max"]) &
                        (lons >= bbox["lon_min"]) & (lons <= bbox["lon_max"])
                    )

                n_pix = int(np.sum(mask))
                if n_pix > 0:
                    county_ts[county]["dates"].append(date_str)
                    county_ts[county]["obs"].append(float(np.mean(obs_raw[mask])))
                    county_ts[county]["pred"].append(float(np.mean(preds_raw[mask])))
                    county_ts[county]["pixel_counts"].append(n_pix)

        # Dynamically size subplot grid to accommodate any number of target counties
        n_counties = len(active_counties)
        ncols = 2 if n_counties > 1 else 1
        nrows = int(np.ceil(n_counties / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7.5 * ncols, 5.0 * nrows),
            sharex=True,
            squeeze=False,
        )
        axes_flat = axes.flatten()

        county_r2 = {}
        county_metrics = {}

        for idx, (county, data) in enumerate(county_ts.items()):
            ax = axes_flat[idx]
            obs_arr = np.array(data["obs"], dtype=float)
            pred_arr = np.array(data["pred"], dtype=float)

            if len(obs_arr) > 1:
                m = _compute_regression_metrics(obs_arr, pred_arr)
            else:
                m = {
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "r2": float("nan"),
                    "pearson_r": float("nan"),
                    "bias": float("nan"),
                }

            m["mean_n_pixels"] = (
                float(np.mean(data["pixel_counts"])) if data["pixel_counts"] else 0.0
            )
            m["n_months"] = int(len(obs_arr))
            m["spatial_masking"] = masking_mode

            county_r2[county] = m["r2"]
            county_metrics[county] = m

            ax.axhspan(0, 20, color="#d73027", alpha=0.10, label="Severe/Extreme (<20)")
            ax.axhspan(20, 35, color="#fc8d59", alpha=0.10, label="Moderate (20-35)")
            ax.plot(data["dates"], obs_arr, "k--", label="Observed VCI3M", linewidth=1.6)
            ax.plot(data["dates"], pred_arr, color="#e66101", label="EA-LSTM Forecast", linewidth=1.8)
            ax.set_title(
                f"{county.upper()} — R²: {m['r2']:.2f} | RMSE: {m['rmse']:.2f} "
                f"(~{int(round(m['mean_n_pixels']))} px)"
            )
            ax.set_ylabel("VCI3M")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.legend(loc="best", fontsize=8)

        for extra_ax in axes_flat[n_counties:]:
            extra_ax.axis("off")

        plt.tight_layout()
        plt.savefig(self.analysis_dir / "county_timeseries.png", dpi=200)
        plt.close()

        with (self.analysis_dir / "county_r2_scores.json").open("w") as f:
            json.dump(county_r2, f, indent=4)
        with (self.analysis_dir / "county_metrics.json").open("w") as f:
            json.dump(county_metrics, f, indent=4)

        print(
            f"County-level evaluation completed ({masking_mode}). "
            f"R2 summary: { {k: round(v, 3) for k, v in county_r2.items()} }"
        )

    def compute_shap_ranking(
        self,
        model: EALSTMNetwork,
        feature_names: Optional[List[str]] = None,
        dynamic_names: Optional[List[str]] = None,
        static_names: Optional[List[str]] = None,
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Computes feature importance rankings using gradient-based attribution
        (Lees et al. Figure 11), grouped and color-coded by feature family.
        """
        import shap

        test_loader = EALSTMDataLoader(
            data_path=self.data_path,
            mode="test",
            batch_size=min(num_samples, 512),
            shuffle=True,
            device=str(self.device),
        )

        x_d_chunks, x_s_chunks = [], []
        collected = 0
        for bx_d, bx_s, _, _ in test_loader.get_batches():
            x_d_chunks.append(bx_d)
            x_s_chunks.append(bx_s)
            collected += bx_d.shape[0]
            if collected >= num_samples:
                break

        x_d = torch.cat(x_d_chunks, dim=0)[:num_samples]
        x_s = torch.cat(x_s_chunks, dim=0)[:num_samples]

        seq_len, dyn_dim = x_d.shape[1], x_d.shape[2]
        stat_dim = x_s.shape[1]
        d_len = seq_len * dyn_dim
        dyn_shape = x_d.shape[1:]

        auto_dyn, auto_stat = self.discover_feature_names(dyn_dim, stat_dim, test_loader=test_loader)
        resolved_dyn, resolved_stat = auto_dyn, auto_stat
        flattened_mode = False

        if dynamic_names is not None and static_names is not None:
            if len(dynamic_names) == dyn_dim and len(static_names) == stat_dim:
                resolved_dyn, resolved_stat = list(dynamic_names), list(static_names)
            else:
                print(
                    f"Warning: Provided dynamic_names ({len(dynamic_names)}) / static_names ({len(static_names)}) "
                    f"do not match model dimensions ({dyn_dim} / {stat_dim}). "
                    f"Using auto-discovered aliased names from NetCDFs instead."
                )
        elif feature_names is not None:
            if len(feature_names) == dyn_dim + stat_dim:
                resolved_dyn = list(feature_names[:dyn_dim])
                resolved_stat = list(feature_names[dyn_dim:])
            elif len(feature_names) == d_len + stat_dim:
                flattened_mode = True
            elif len(feature_names) == dyn_dim:
                resolved_dyn = list(feature_names)
                resolved_stat = []
            else:
                print(
                    f"Warning: Provided feature_names ({len(feature_names)}) does not match model "
                    f"dimensions ({dyn_dim} dynamic + {stat_dim} static = {dyn_dim + stat_dim}). "
                    f"Using auto-discovered aliased names from NetCDFs instead."
                )

        class ModelWrapper(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, flat_inputs):
                xd = flat_inputs[:, :d_len].reshape(-1, *dyn_shape)
                xs = flat_inputs[:, d_len:]
                return self.net(xd, xs)

        wrapper = ModelWrapper(model).to(self.device)
        wrapper.eval()

        flat_x = torch.cat([x_d.reshape(x_d.shape[0], -1), x_s], dim=1)
        half = flat_x.shape[0] // 2
        if half < 1:
            raise ValueError(f"Need at least 2 samples for SHAP attribution, got {flat_x.shape[0]}.")

        explainer = shap.GradientExplainer(wrapper, flat_x[:half])
        try:
            shap_values = explainer.shap_values(flat_x[half:], rseed=42)
        except TypeError:
            shap_values = explainer.shap_values(flat_x[half:])

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
            shap_values = shap_values.squeeze(-1)

        mean_abs_flat = np.mean(np.abs(shap_values), axis=0)
        dyn_by_lag = mean_abs_flat[:d_len].reshape(seq_len, dyn_dim)
        dyn_shap = dyn_by_lag.mean(axis=0)
        stat_shap = mean_abs_flat[d_len:]

        if flattened_mode and feature_names is not None:
            mean_abs_shap = mean_abs_flat
            plot_names = list(feature_names)
            plot_categories = [
                _classify_feature_category(n, is_static=(idx >= d_len))
                for idx, n in enumerate(plot_names)
            ]
        elif len(resolved_stat) == 0:
            mean_abs_shap = dyn_shap
            plot_names = resolved_dyn
            plot_categories = [_classify_feature_category(n, is_static=False) for n in plot_names]
        else:
            mean_abs_shap = np.concatenate([dyn_shap, stat_shap])
            plot_names = resolved_dyn + resolved_stat
            plot_categories = (
                [_classify_feature_category(n, is_static=False) for n in resolved_dyn]
                + [_classify_feature_category(n, is_static=True) for n in resolved_stat]
            )

        np.save(self.analysis_dir / "mean_abs_shap.npy", mean_abs_shap)

        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        ranking = {plot_names[i]: float(mean_abs_shap[i]) for i in sorted_idx}
        detailed_ranking = [
            {
                "rank": rank + 1,
                "feature": plot_names[i],
                "category": plot_categories[i],
                "mean_abs_shap": float(mean_abs_shap[i]),
            }
            for rank, i in enumerate(sorted_idx)
        ]

        cat_summary: Dict[str, Dict[str, float]] = {}
        total_shap = float(np.sum(mean_abs_shap)) or 1.0
        for cat in sorted(set(plot_categories)):
            vals = [mean_abs_shap[i] for i, c in enumerate(plot_categories) if c == cat]
            cat_sum = float(np.sum(vals))
            cat_summary[cat] = {
                "count": len(vals),
                "total_shap": cat_sum,
                "share_pct": round(100.0 * cat_sum / total_shap, 2),
                "mean_shap": float(np.mean(vals)),
            }

        lag_labels = [f"t-{seq_len - lag}" for lag in range(seq_len)]
        lag_breakdown = {
            lag_labels[lag]: {
                "total_dynamic_shap": float(np.sum(dyn_by_lag[lag])),
                "top_features": {
                    resolved_dyn[j]: float(dyn_by_lag[lag, j])
                    for j in np.argsort(dyn_by_lag[lag])[::-1][:10]
                },
            }
            for lag in range(seq_len)
        }

        with (self.analysis_dir / "shap_ranking.json").open("w") as f:
            json.dump(detailed_ranking, f, indent=4)
        with (self.analysis_dir / "shap_category_summary.json").open("w") as f:
            json.dump(cat_summary, f, indent=4)
        with (self.analysis_dir / "shap_lag_breakdown.json").open("w") as f:
            json.dump(lag_breakdown, f, indent=4)

        fig_width = max(12.0, 0.32 * len(sorted_idx))
        plt.figure(figsize=(fig_width, 6.5))
        bar_colors = [CATEGORY_COLORS.get(plot_categories[i], "#1f78b4") for i in sorted_idx]
        plt.bar(range(len(sorted_idx)), mean_abs_shap[sorted_idx], color=bar_colors)
        plt.xticks(range(len(sorted_idx)), [plot_names[i] for i in sorted_idx], rotation=90, fontsize=8)
        plt.ylabel("Mean Absolute Shapley Value")
        plt.title(f"EA-LSTM Feature Importance Ranking (n={flat_x.shape[0] - half} test samples)")

        present_cats = [c for c in CATEGORY_COLORS if c in set(plot_categories)]
        legend_handles = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in present_cats]
        plt.legend(handles=legend_handles, title="Feature Group", loc="upper right")

        plt.tight_layout()
        plt.savefig(self.analysis_dir / "shap_importance_ranking.png", dpi=200)
        plt.close()

        top_5 = [(plot_names[i], round(float(mean_abs_shap[i]), 4)) for i in sorted_idx[:5]]
        print("SHAP feature ranking generated successfully. Top 5:", top_5)
        return ranking

    def run_full_evaluation(
        self,
        model: EALSTMNetwork,
        feature_names: Optional[List[str]] = None,
        dynamic_names: Optional[List[str]] = None,
        static_names: Optional[List[str]] = None,
        num_shap_samples: int = 100,
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """Runs holdout test metrics, regional county evaluation, clustering, and SHAP in one call."""
        test_results = self.evaluate_test_set(model)
        shap_ranking = self.compute_shap_ranking(
            model=model,
            feature_names=feature_names,
            dynamic_names=dynamic_names,
            static_names=static_names,
            num_samples=num_shap_samples,
        )
        return {"test_results": test_results, "shap_ranking": shap_ranking}