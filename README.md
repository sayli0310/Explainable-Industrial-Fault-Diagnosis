# Explainable Industrial Fault Diagnosis

## 1. Project Overview

Explainable Industrial Fault Diagnosis is a machine learning-based system designed to identify bearing faults using vibration signals.

The project uses a Random Forest classifier to distinguish between four bearing conditions and SHAP (SHapley Additive exPlanations) to explain the model's predictions.

A Streamlit web application allows users to upload MATLAB vibration signal files and obtain fault predictions along with feature-based explanations.

## 2. Objectives

- Analyze industrial bearing vibration signals.
- Perform exploratory data analysis and signal processing.
- Extract statistical features from vibration signals.
- Train a machine learning model for fault classification.
- Evaluate model performance.
- Interpret predictions using SHAP explainability.
- Develop an interactive Streamlit application.

## 3. Dataset

The project uses the Case Western Reserve University (CWRU) Bearing Dataset.

Four bearing conditions are considered:

1. Normal
2. Inner Race Fault
3. Ball Fault
4. Outer Race Fault

The dataset contains 16 MATLAB recordings, with four recordings representing each condition.

Drive End vibration signals are used for feature extraction.

## 4. Methodology

### Step 1: Exploratory Data Analysis
- Dataset inspection
- Statistical summaries
- Missing-value analysis
- Class distribution
- Feature histograms and boxplots
- Correlation analysis

### Step 2: Signal Processing
- Time-domain signal visualization
- Fast Fourier Transform (FFT)
- Frequency-domain analysis
- Butterworth filtering

### Step 3: Feature Extraction

Each vibration signal is divided into non-overlapping segments of 2048 samples.

Ten statistical features are extracted:

1. Mean
2. Standard Deviation
3. RMS
4. Variance
5. Maximum
6. Minimum
7. Peak-to-Peak
8. Skewness
9. Kurtosis
10. Crest Factor

### Step 4: Model Training

A Random Forest Classifier is trained to classify the four bearing conditions.

### Step 5: Model Evaluation

Model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Recordings are separated at file level during training and testing to reduce data leakage between vibration segments from the same recording.

### Step 6: Explainable AI

SHAP is used to interpret model predictions.

The explanations identify which statistical features contribute positively or negatively to individual predictions.

## 5. Technology Stack

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP
- Streamlit
- Joblib

## 6. Project Structure

```text
projectml1/
│
├── app/
│   └── app.py
│
├── dataset/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_signal_processing.ipynb
│   ├── 04_model_training.ipynb
│   └── 06_explainability.ipynb
│
├── src/
│   ├── __init__.py
│   └── feature_extraction.py
│
├── results/
│   ├── eda/
│   └── models/
│       ├── random_forest_model.pkl
│       └── label_encoder.pkl
│
├── requirements.txt
└── README.md
```

## 7. Installation

Clone the repository:

```bash
git clone 
```

Navigate to the project directory:

```bash
cd projectml1
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## 8. Running the Application

From the main project directory, execute:

```bash
python -m streamlit run app/app.py
```

The application will open in your web browser.

Upload a supported MATLAB (.mat) vibration signal file to obtain:

- Extracted statistical features
- Predicted bearing condition
- Prediction probabilities
- SHAP-based prediction explanation

## 9. Key Highlights

- Machine learning-based industrial fault classification.
- Statistical feature extraction from vibration signals.
- Frequency-domain signal analysis.
- Explainable AI using SHAP.
- Interactive Streamlit interface.
- File-level train-test separation to reduce data leakage.

## 10. Future Improvements

- Explore additional machine learning algorithms.
- Improve model generalization using more independent recordings.
- Incorporate deep learning-based fault diagnosis.
- Add real-time vibration monitoring.
- Deploy the application to a cloud platform.

## 11. Conclusion

This project demonstrates how machine learning and explainable artificial intelligence can be combined to develop an interpretable industrial bearing fault diagnosis system.

It provides an interactive approach to identifying bearing conditions from vibration signals while offering explanations for the model's predictions.