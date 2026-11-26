import os
import sys
from pathlib import Path
from src.preprocessor.utils import check_raw_files
from src.preprocessor import (
    CHIRPSPreprocessor,
    MODISNDVIPreprocessor,
    SRTMPreprocessor,
    ERA5LandPreprocessor,
    ERA5LandSoilTypePreprocessor,
)

# Usage: python run_pipeline.py <data_folder> <task: preprocess, engineer, ealstm, analysis> <dataset_name: optional> <regrid: optional>

tasks = ["preprocess", "engineer", "ealstm", "analysis"]
datasets = ["chirps", "modis_ndvi", "srtm", "era5_land", "era5_land_slt"]

# Check command-line arguments
data_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
task = sys.argv[2] if len(sys.argv) > 2 else None
dataset_name = sys.argv[3] if len(sys.argv) > 3 else None
regrid = Path(sys.argv[4]) if len(sys.argv) > 4 else None

if task and task not in tasks:
    print(f"Invalid task: {task}. Please choose from {tasks}")
    sys.exit(1)

if dataset_name and dataset_name not in datasets:
    print(f"Invalid dataset name: {dataset_name}. Please choose from {datasets}")
    sys.exit(1)

if data_folder and not Path(data_folder).exists():
    print(f"Data path does not exist: {data_folder}")
    sys.exit(1)

if regrid and not regrid.exists():
    print(f"Regrid path does not exist: {regrid}")
    sys.exit(1)

def preprocess(data_folder, dataset_name):
    if dataset_name == "chirps":
        preprocessor = CHIRPSPreprocessor(data_folder=data_folder)
        check_raw_files(dataset_name, preprocessor.raw_folder / dataset_name,
                         preprocessor.gee_exports / dataset_name)
        output_path = preprocessor.preprocess(
            subset_str="kenya",
            regrid=regrid,
            regrid_method="bilinear" if regrid else None,
            n_processes=1 if regrid else len(os.sched_getaffinity(0)),
            cleanup=False,
        )
    elif dataset_name == "modis_ndvi":
        preprocessor = MODISNDVIPreprocessor(data_folder=data_folder)
        check_raw_files(
            preprocessor.dataset,
            preprocessor.raw_folder / preprocessor.dataset,
            preprocessor.gee_exports / preprocessor.dataset,
        )
        output_path = preprocessor.preprocess(
            subset_str="kenya",
            regrid=regrid,
            regrid_method="bilinear" if regrid else None,
            rolling_window=3,
            reference_period=('2002', '2022'),
            cleanup=False,
        )
    elif dataset_name == "srtm":
        preprocessor = SRTMPreprocessor(data_folder=data_folder)
        check_raw_files(
            dataset_name,
            preprocessor.raw_folder / dataset_name,
            preprocessor.gee_exports / dataset_name,
        )
        output_path = preprocessor.preprocess(
            subset_str="kenya",
            regrid=regrid,
            regrid_method="bilinear" if regrid else None,
            clean_weights=False,
        )
    elif dataset_name == "era5_land":
        preprocessor = ERA5LandPreprocessor(data_folder=data_folder)
        check_raw_files(
            preprocessor.dataset,
            preprocessor.raw_folder / preprocessor.dataset,
            preprocessor.gee_exports / preprocessor.dataset,
        )
        output_path = preprocessor.preprocess(
            subset_str="kenya",
            regrid=regrid,
            regrid_method="bilinear" if regrid else None,
            n_processes=1 if regrid else len(os.sched_getaffinity(0)),
            cleanup=False,
        )
    elif dataset_name == "era5_land_slt":
        preprocessor = ERA5LandSoilTypePreprocessor(data_folder=data_folder)
        check_raw_files(
            preprocessor.dataset,
            preprocessor.raw_folder / preprocessor.dataset,
            preprocessor.gee_exports / preprocessor.dataset,
        )
        output_path = preprocessor.preprocess(
            subset_str="kenya",
            regrid=regrid,
            regrid_method="nearest_s2d" if regrid else None,
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    print(f"Preprocessing completed for {dataset_name}. Output saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    if not data_folder or not task:
        print("Usage: python run_pipeline.py <data_folder> <task: preprocess, engineer, ealstm, analysis> <dataset_name: optional> <regrid: optional>")
        sys.exit(1)

    if task == "preprocess":
        if not dataset_name:
            print("Please specify a dataset name for preprocessing.")
            sys.exit(1)
        preprocess(data_folder, dataset_name)
    else:
        print(f"Task '{task}' is not implemented yet.")
