# -------------------------------------------------------------
# 1. MOUNT DRIVE AND RUN PRE-FLIGHT TEST
# -------------------------------------------------------------
from google.colab import drive
import os

drive.mount('/content/drive')
REPO_DIR = '/content/drive/MyDrive/ea_lstm_drought'
%cd $REPO_DIR

!python -m tests.test_pipeline

# -------------------------------------------------------------
# 2. RUN ENGINEER (2023 VALIDATION, 2016-2018 TEST)
# -------------------------------------------------------------
from src.engineer.engineer import EALSTMDataEngineer

engineer = EALSTMDataEngineer(data_folder="data", process_static=True)
engineer.engineer(
    test_years=,
    val_years=2023,
    target_variable="modis_VCI3M",
    pred_months=3,
    expected_length=3,
    pixel_means=True,
    global_means=False
)

# -------------------------------------------------------------
# 3. TRAIN EA-LSTM (UNWEIGHTED LOSS, BEST WEIGHT CHECKPOINTING)
# -------------------------------------------------------------
from src.models.ealstm import EARecurrentModel

trainer = EARecurrentModel(
    data_path="data",
    hidden_size=64,
    rnn_dropout=0.25,
    learning_rate=1e-3
)

trainer.train(
    num_epochs=100,
    patience=10,
    batch_size=256
)

# -------------------------------------------------------------
# 4. RUN COMPREHENSIVE EVALUATION & SHAP ANALYSIS
# -------------------------------------------------------------
from src.analysis.evaluation_vci import ModelEvaluator

evaluator = ModelEvaluator(data_path="data", target_variable="modis_VCI3M")
evaluator.evaluate_test_set(trainer.model)