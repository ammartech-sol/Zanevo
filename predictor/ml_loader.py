import joblib
import json
from pathlib import Path
import numpy as np
import onnxruntime

BASE = Path(__file__).resolve().parent.parent / 'ml_models'

try:
    heart = {
        'model':   joblib.load(BASE / 'heart_model.pkl'),
        'scaler':  joblib.load(BASE / 'heart_scaler.pkl'),
        'columns': joblib.load(BASE / 'heart_columns.pkl'),
    }
    print("Heart model loaded OK")
except Exception as e:
    print(f"Heart model FAILED: {e}")

try:
    churn = {
        'model':     joblib.load(BASE / 'churn_model.pkl'),
        'encoder':   joblib.load(BASE / 'churn_encoder.pkl'),
        'columns':   joblib.load(BASE / 'churn_columns.pkl'),
        'threshold': joblib.load(BASE / 'churn_threshold.pkl'),
    }
    print("Churn model loaded OK")
except Exception as e:
    print(f"Churn model FAILED: {e}")

try:
    attrition = {
        'model':     joblib.load(BASE / 'xgboost_ibm_attrition.pkl'),
        'encoder':   joblib.load(BASE / 'attrition_ohe_encoder.pkl'),
        'columns':   joblib.load(BASE / 'attrition_columns.pkl'),
        'threshold': joblib.load(BASE / 'attrition_threshold.pkl'),
    }
    print("Attrition model loaded OK")
except Exception as e:
    print(f"Attrition model FAILED: {e}")

try:
    census = {
        'model':     joblib.load(BASE / 'xgboost_adult_census.pkl'),
        'encoder':   joblib.load(BASE / 'adult_census_encoder.pkl'),
        'columns':   joblib.load(BASE / 'adult_census_columns.pkl'),
        'threshold': joblib.load(BASE / 'adult_census_threshold.pkl'),
    }
    print("Adult_census model loaded OK")
except Exception as e:
    print(f"Adult_census model FAILED: {e}")

try:
    ames = {
        'model':     joblib.load(BASE / 'ridge_ames.pkl'),
        'preprocessor':   joblib.load(BASE / 'ames_preprocessor.pkl'),
        'columns':   joblib.load(BASE / 'ames_columns.pkl'),
    }
    print("Ames_housing model loaded OK")
except Exception as e:
    print(f"Ames_housing model FAILED: {e}")

try:
    student = {
        'model':     joblib.load(BASE / 'student_performance_model.pkl'),
        'columns':   joblib.load(BASE / 'student_columns.pkl'),
    }
    print("Student_Prediction model loaded OK")
except Exception as e:
    print(f"Student_Prediction model FAILED: {e}")


try:
    with open(BASE / 'commerce_cluster_labels.json', 'r') as f:
        _cluster_labels = json.load(f)

    segmentation = {
        'model':  joblib.load(BASE / 'commerce_kmeans_model.pkl'),
        'scaler': joblib.load(BASE / 'commerce_scaler.pkl'),
        'labels': _cluster_labels,
    }
    print("Segmentation model loaded OK")
except Exception as e:
    print(f"Segmentation model FAILED: {e}")

def load_mbti_dim(letter):
    return {
        'W1': np.load(BASE / f'{letter}_W1.npy'),
        'b1': np.load(BASE / f'{letter}_b1.npy'),
        'W2': np.load(BASE / f'{letter}_W2.npy'),
        'b2': np.load(BASE / f'{letter}_b2.npy'),
        'W3': np.load(BASE / f'{letter}_W3.npy'),
        'b3': np.load(BASE / f'{letter}_b3.npy'),
    }

try:
    mbti = {
        'IE': load_mbti_dim('IE'),
        'NS': load_mbti_dim('NS'),
        'TF': load_mbti_dim('TF'),
        'JP': load_mbti_dim('JP'),
    }
    print("MBTI loaded OK")
except Exception as e:
    print(f"MBTI FAILED: {e}")

try:
    _plant_disease_opts = onnxruntime.SessionOptions()
    _plant_disease_opts.intra_op_num_threads = 1
    _plant_disease_opts.inter_op_num_threads = 1
    plant_disease_session = onnxruntime.InferenceSession(
        str(BASE / 'plant_disease_unet_quantized.onnx'),
        sess_options=_plant_disease_opts
    )
    print("PlantDisease ONNX model loaded OK")
except Exception as e:
    plant_disease_session = None
    print(f"PlantDisease FAILED: {e}")