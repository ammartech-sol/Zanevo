# Zanevo

A unified ML prediction platform serving production-trained models across healthcare, business, real estate, and education. Built with Django and deployed on Render.

---

## Models

**Classification (4 models)**

| Model | Algorithm | Dataset | Key Metric |
|---|---|---|---|
| Customer Churn | Random Forest | Telco (Kaggle) | F1 0.63, Threshold 0.323 |
| Employee Attrition | XGBoost | IBM HR Dataset | Threshold 0.30 |
| Heart Disease Risk | Logistic Regression | Cleveland (UCI) | 13 features |
| Adult Census Income | XGBoost | UCI Adult Census | F1 0.72, Threshold 0.40 |

**Regression (2 models)**

| Model | Algorithm | Dataset | Key Metric |
|---|---|---|---|
| Ames House Price | Ridge Regression | Ames Housing | R² 0.9315, MAE $11,004 |
| Student Performance | Random Forest | Student Performance | R² ~0.99 |

**Clustering (1 model)**

| Model | Algorithm | Dataset | Key Metric |
|---|---|---|---|
| Customer Segmentation | K-Means (K=3) | Olist Brazilian E-Commerce | Silhouette 0.4129 |

---

## Technical Decisions

**Threshold tuning on imbalanced classifiers**
Churn (0.323), Attrition (0.30), and Census Income (0.40) all use custom thresholds below 0.5. The default threshold optimizes accuracy, which is the wrong objective when missing a churner costs more than a false alarm. Lowering the threshold shifts the tradeoff toward catching more positives.

**Ridge over LightGBM for Ames Housing**
LightGBM showed a 6.4-point train/test gap (0.9826 vs 0.9183) without tuning, a sign of overfitting. Ridge with L2 regularization handled the 167-feature post-encoding space better out of the box.

**Algorithm selection followed the data**
Heart Disease uses Logistic Regression because the Cleveland dataset is only 303 rows and clinical interpretability matters more than marginal accuracy gains. Attrition uses XGBoost because HR data has complex feature interactions that linear models structurally miss.

**Feature engineering validated by importance scores**
Six features were engineered for the Attrition model including `IsOverworked`, combining overtime status with poor work-life balance as a stress signal. It ranked in the top 5 most important features out of 35, confirming the engineering decision rather than just adding noise.

**F1 over accuracy on imbalanced datasets**
The UCI Adult dataset is 76/24 class-imbalanced. Predicting everyone as <=50K hits 76% accuracy while being completely useless. The IBM HR dataset is 84/16. In both cases F1 and recall are the meaningful metrics; accuracy is a number that hides a broken model.

**Dropping zero-variance features before clustering**
In the Olist segmentation project, Frequency showed that over 90% of customers had placed exactly one order. After log transform and outlier removal it had a standard deviation of 0.0, meaning it could not separate any customers from each other. A feature that adds no signal only distorts centroid positions. It was dropped.

---

## Feature Engineering

Each model has engineered features that are computed automatically from user inputs and never entered directly.

| Model | Engineered Features |
|---|---|
| Churn | `is_month_to_month`, `avg_monthly_spend`, `total_services` |
| Attrition | `AvgSatisfaction`, `IncomePerYearExp`, `StagnationRatio`, `IsLoyal`, `IsOverworked`, `CareerGrowthRate` |
| Census Income | `age_edu` (age x education level), `has_capital` (any capital activity flag) |
| Ames Housing | `TotalSF`, `TotalBathrooms`, `HouseAge`, `RemodAge` |

---

## Stack

- Django 5.2
- scikit-learn 1.7.2, XGBoost 3.2.0
- pandas, numpy, scipy
- WhiteNoise for static file serving
- Gunicorn as the WSGI server
- Deployed on Render

---

## Folder Structure

```
predictive_portal/
├── portal/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── predictor/
│   ├── views.py
│   ├── ml_loader.py
│   ├── models.py
│   └── urls.py
├── templates/
│   ├── home.html
│   ├── churn.html
│   ├── attrition.html
│   ├── heart.html
│   ├── census.html
│   ├── ames.html
│   ├── student.html
│   ├── segmentation.html
│   ├── privacy.html
│   └── terms.html
├── ml_models/
│   └── *.pkl
├── .env
├── .gitignore
├── manage.py
├── requirements.txt
└── README.md
```

---

## Local Setup

1. Clone the repo
2. Create and activate a conda environment
```bash
conda create -n ai-course python=3.10
conda activate ai-course
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Create a `.env` file in the project root
```
SECRET_KEY=your-secret-key-here
DEBUG=True
```
5. Add the `ml_models/` folder with all `.pkl` files (not tracked by git due to file size)
6. Collect static files
```bash
python manage.py collectstatic
```
7. Run the server
```bash
python manage.py runserver
```

---

## Saved Model Files

| File | Description |
|---|---|
| `churn_model.pkl` | Random Forest, F1-optimized params |
| `churn_encoder.pkl` | Fitted ColumnTransformer |
| `churn_threshold.pkl` | Optimal threshold (0.323) |
| `churn_columns.pkl` | Ordered feature columns |
| `xgboost_ibm_attrition.pkl` | XGBoost attrition classifier |
| `attrition_ohe_encoder.pkl` | One-hot encoder for HR features |
| `attrition_threshold.pkl` | Optimal threshold (0.30) |
| `heart_model.pkl` | Logistic Regression classifier |
| `heart_scaler.pkl` | StandardScaler for clinical inputs |
| `xgboost_adult_census.pkl` | XGBoost income classifier |
| `adult_census_threshold.pkl` | Optimal threshold (0.40) |
| `ridge_ames.pkl` | Ridge Regression price model |
| `ames_preprocessor.pkl` | Full preprocessing pipeline |
| `student_performance_model.pkl` | Random Forest regression |
| `commerce_kmeans_model.pkl` | K-Means clustering (K=3) |
| `commerce_scaler.pkl` | Scaler for RFM features |
| `commerce_cluster_labels.json` | Cluster ID to segment name map |

---

## Requirements

```
django==5.2.14
gunicorn==26.0.0
whitenoise==6.12.0
python-dotenv==1.2.2
pandas==2.3.3
numpy==2.2.6
scikit-learn==1.7.2
xgboost==3.2.0
joblib==1.5.3
scipy==1.15.2
```
