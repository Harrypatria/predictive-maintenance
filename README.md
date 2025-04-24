# 🔧 Predictive Maintenance with Machine Learning

<div align="center">
  
![Predictive Maintenance](https://img.shields.io/badge/Predictive-Maintenance-blue?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABhGlDQ1BJQ0MgcHJvZmlsZQAAKJF9kT1Iw0AcxV9TpSIVBzuIOGSogmBBVMRRq1CECqFWaNXB5NIvaNKQpLg4Cq4FBz8Wqw4uzro6uAqC4AeIm5uToouU+L+k0CLWg+N+vLv3uHsHCPUy06yOcUDTbTOViIuZ7KoYeEUQfehBCCMys4xZSUohjF3f8fXjdRbPsz/350pqzmKATySeZYZpE28QT2/aOud94jArSQrxOfGYQRckfuS67PIb56LDAs8MG5nUPHGYWCx2sNzBrGSoxFPEEUXVKF/Iuqxw3uKsVmqsdU/+wmBeW0lzneYw4lhCAkmIkFFFCWVYiNGqkWIiRftxD/+w40+SSyZXCYwcC6hAheTo4aewfpv9wMDZpUhc14rF496TQgDnF8l4nWPg5g54PPj4pC3Jkl8tUAwIvhn9/TGM+W5QXLkQ8zLwdGdHyR5A/wxcPBdl670+3d3b59t/TjvyR2ACvdvNQYcCAQVoAAAAlElEQVQ4jWNgoBX4PzmA/39SxH8QjVPzgU0O/3GZArMExkZWTLQBMM0gA/5vVfj/f4fy//9rlVE0wx1CVAjgjAVkG/5vU4IbAlMD0gzTgxKIIA3/typidwUWzUQZQFAziWFClCtIdgVBA5A1Y3MFPheQ5AqCBuDSjM0VmEGMVQNBA2CaQYmFkMtBmrEBrBqQQwmXZgDzHyKXz9VIzAAAAABJRU5ErkJggg==)

[![GitHub stars](https://img.shields.io/github/stars/Harrypatria/predictive-maintenance?style=social)](https://github.com/Harrypatria/predictive-maintenance/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Harrypatria/predictive-maintenance?style=social)](https://github.com/Harrypatria/predictive-maintenance/network/members)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  <img src="https://j.gifs.com/76kDrQ.gif" alt="animated gif">
</p>

## 🌟 Overview

A comprehensive implementation of predictive maintenance using machine learning, designed for industrial equipment and IoT applications. This repository demonstrates how to predict equipment failures before they occur, optimize maintenance schedules, and reduce downtime.

The project includes data preprocessing, feature engineering, model training, evaluation, and deployment strategies, all with real-world industrial examples and optimization techniques.

## ✨ Key Features

- **📊 Advanced Data Processing**: Techniques for handling time-series sensor data
- **🧮 Feature Engineering**: Creating meaningful features from raw sensor data
- **🤖 Multiple ML Models**: Comparison of various algorithms for failure prediction
- **⚡ Performance Optimization**: Parameter tuning and model selection
- **📉 Time to Failure (TTF) Prediction**: Estimating remaining useful life of components
- **🔄 Anomaly Detection**: Identifying unusual patterns that may indicate failures
- **📱 Production Integration**: APIs and implementation strategies for real-world use
- **📈 Interactive Visualization**: Dashboards for monitoring and analysis

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Harrypatria/predictive-maintenance.git
cd predictive-maintenance

# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook
```

## 📋 Project Structure

1. **Data Collection and Preparation**
   - Sensor data processing
   - Feature extraction
   - Time series handling
   - Handling imbalanced data

2. **Exploratory Data Analysis**
   - Sensor correlations
   - Pattern identification
   - Failure mode analysis
   - Statistical insights

3. **Feature Engineering**
   - Signal processing techniques
   - Statistical features
   - Time-domain features
   - Frequency-domain features

4. **Model Development**
   - Classification models
   - Regression for TTF prediction
   - Anomaly detection
   - Ensemble methods

5. **Evaluation and Optimization**
   - Cross-validation strategies
   - Hyperparameter tuning
   - Metric selection
   - Model comparison

6. **Deployment and Integration**
   - Real-time scoring
   - Alert system
   - Maintenance scheduling
   - API development

## 🔥 Models and Techniques

| Model Type | Implementation | Use Case |
|------------|----------------|----------|
| Random Forest | Failure classification | Binary classification of failing/non-failing equipment |
| Gradient Boosting | Remaining useful life | Regression to predict time until failure |
| LSTM | Sequence prediction | Time-series analysis of sensor patterns |
| Isolation Forest | Anomaly detection | Identifying unusual equipment behavior |
| PCA/Autoencoders | Dimensionality reduction | Feature extraction from high-dimensional sensor data |
| XGBoost | Production prediction | High-performance implementation for deployment |

## 💡 Real-world Applications

- **Manufacturing**: Predicting machine failures in production lines
- **Energy**: Monitoring wind turbine performance and maintenance needs
- **Healthcare**: Medical equipment reliability and service scheduling
- **Transportation**: Fleet management and vehicle component monitoring
- **HVAC Systems**: Predicting failures in heating and cooling equipment
- **IT Infrastructure**: Server and network equipment monitoring

## 📈 Performance Metrics

The models in this repository are evaluated using multiple metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of correct positive predictions to total positive predictions
- **Recall**: Ratio of correct positive predictions to all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **MAE/RMSE**: For regression models predicting time-to-failure
- **Economic Value**: Cost savings from prevented downtime

## 🧪 Example: Predicting Equipment Failure

```python
# Load and preprocess sensor data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load sensor data
df = pd.read_csv('sensor_data.csv')

# Feature engineering
df['rolling_mean'] = df.groupby('machine_id')['temperature'].transform(lambda x: x.rolling(window=24).mean())
df['rolling_std'] = df.groupby('machine_id')['temperature'].transform(lambda x: x.rolling(window=24).std())
df['pressure_temp_ratio'] = df['pressure'] / df['temperature']
df = df.dropna()

# Prepare features and target
features = ['temperature', 'pressure', 'vibration', 'rotation_speed', 
            'rolling_mean', 'rolling_std', 'pressure_temp_ratio']
X = df[features]
y = df['failure_within_24h']  # Binary target: Will fail within 24 hours?

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("Feature importance:")
print(importance)
```

## 🔄 Implementation Pipeline

![Implementation Pipeline](https://img.shields.io/badge/Pipeline-Implementation-blue?style=for-the-badge)

1. **Data Collection**
   - Gather sensor data from equipment
   - Store in appropriate database (time-series optimized)
   - Implement data validation and cleaning

2. **Feature Processing**
   - Extract relevant features
   - Handle missing values and outliers
   - Normalize/standardize features

3. **Model Training**
   - Split data into training/validation/test sets
   - Train multiple model types
   - Compare performance

4. **Deployment**
   - Create API for real-time scoring
   - Implement threshold-based alerts
   - Integrate with maintenance management systems

5. **Monitoring and Retraining**
   - Track model performance over time
   - Collect feedback from maintenance actions
   - Retrain models periodically with new data

## 🛠️ Advanced Techniques

<details>
<summary><b>Handling Imbalanced Data</b></summary>

Most maintenance datasets are highly imbalanced (failures are rare events):

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Define resampling strategy
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)

# Create pipeline with resampling and model
pipeline = Pipeline([
    ('over', over),
    ('under', under),
    ('model', RandomForestClassifier())
])

# Fit pipeline
pipeline.fit(X_train, y_train)
```
</details>

<details>
<summary><b>Time-Series Feature Extraction</b></summary>

Creating meaningful features from time-series sensor data:

```python
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

# Extract features
extracted_features = extract_features(df, column_id="machine_id", column_sort="timestamp")

# Remove NaN values
impute(extracted_features)

# Select relevant features
filtered_features = select_features(extracted_features, y_train)
```
</details>

<details>
<summary><b>Remaining Useful Life (RUL) Prediction</b></summary>

Predicting the time until failure:

```python
import xgboost as xgb

# Prepare data with time-to-failure as target 
X_rul = df[features]
y_rul = df['days_to_failure']

# Train XGBoost regressor
rul_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)
rul_model.fit(X_rul, y_rul)

# Predict remaining useful life
predicted_rul = rul_model.predict(X_test)
```
</details>

## 📊 Visualization Examples

The notebooks include various visualizations:

- Component degradation over time
- Feature importance charts
- Confusion matrices
- ROC curves
- Predicted vs. actual failure times
- Real-time monitoring dashboards
- Health index trends

## 🤔 Why Predictive Maintenance?

- **Cost Reduction**: Up to 30% lower maintenance costs
- **Downtime Prevention**: 70% reduction in breakdowns
- **Extended Equipment Life**: 20-40% longer machine lifespans
- **Improved Safety**: Fewer catastrophic failures
- **Optimized Inventory**: Better spare parts management
- **Enhanced Planning**: Data-driven maintenance scheduling
- **Higher Productivity**: Increased overall equipment effectiveness (OEE)

## 🔧 Troubleshooting Common Issues

<details>
<summary><b>Dealing with Missing Sensor Data</b></summary>

When sensors fail or provide inconsistent data:

```python
# Method 1: Forward fill with a limit
df.fillna(method='ffill', limit=3, inplace=True)

# Method 2: Use rolling windows for imputation
for column in sensor_columns:
    mask = df[column].isna()
    df.loc[mask, column] = df[column].rolling(window=24, min_periods=1).mean()

# Method 3: Use machine learning for imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[sensor_columns] = imputer.fit_transform(df[sensor_columns])
```
</details>

<details>
<summary><b>Model Drift and Retraining</b></summary>

Models can degrade over time as equipment or conditions change:

```python
from sklearn.metrics import precision_score
import datetime

def check_model_drift(model, X_recent, y_recent, threshold=0.7):
    """Monitor model performance and retrain if needed"""
    # Predict on recent data
    y_pred = model.predict(X_recent)
    
    # Calculate precision (or other relevant metric)
    current_precision = precision_score(y_recent, y_pred)
    
    # Log performance
    with open('model_performance_log.csv', 'a') as f:
        f.write(f"{datetime.datetime.now()},{current_precision}\n")
    
    # Check if retraining is needed
    if current_precision < threshold:
        print(f"Model performance below threshold ({current_precision} < {threshold})")
        print("Retraining model with new data...")
        
        # Get training data including recent data
        X_new_train = pd.concat([X_train, X_recent])
        y_new_train = pd.concat([y_train, y_recent])
        
        # Retrain model
        model.fit(X_new_train, y_new_train)
        
        # Save new model
        save_model(model, f"model_retrained_{datetime.datetime.now().strftime('%Y%m%d')}.pkl")
        
        return True  # Model was retrained
    
    return False  # No retraining needed
```
</details>

<details>
<summary><b>Handling Multiple Failure Modes</b></summary>

Real equipment can fail in many different ways:

```python
# Multi-class classification for different failure types
from sklearn.multiclass import OneVsRestClassifier

# Prepare multi-class target (failure types instead of binary)
y_failure_types = df['failure_type']  # e.g., 'bearing', 'electrical', 'hydraulic', etc.

# Train multi-class model
multiclass_model = OneVsRestClassifier(RandomForestClassifier())
multiclass_model.fit(X_train, y_failure_types)

# Predict failure types
predicted_failure_types = multiclass_model.predict(X_test)
```
</details>

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

<div align="center">

## 🌟 Support This Project
**Follow me on GitHub**: [![GitHub Follow](https://img.shields.io/github/followers/Harrypatria?style=social)](https://github.com/Harrypatria?tab=followers)
**Star this repository**: [![GitHub Star](https://img.shields.io/github/stars/Harrypatria/predictive-maintenance?style=social)](https://github.com/Harrypatria/predictive-maintenance/stargazers)
**Connect on LinkedIn**: [![LinkedIn Follow](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harry-patria/)

Click the buttons above to show your support!

</div>
