# 🏃 Human Activity Recognition (HAR)

A machine learning pipeline that classifies human activities (walking, sitting, standing, etc.) from smartphone sensor data using the **UCI HAR Dataset** (561 features from accelerometer and gyroscope signals).

---

## 📌 Problem Statement

Identify 6 human activities in real-time from waist-worn smartphone sensor readings — enabling applications in healthcare monitoring, fitness tracking, and elderly care.

**Activities:** `LAYING · SITTING · STANDING · WALKING · WALKING_UPSTAIRS · WALKING_DOWNSTAIRS`

---

## 📂 Project Structure

```
har/
├── har_train.py          # Full training pipeline
├── app.py                # Streamlit inference app
├── har_model.pkl         # Saved best model
├── scaler.pkl            # Fitted StandardScaler
├── label_encoder.pkl     # LabelEncoder for activities
├── feature_names.pkl     # List of 561 feature names
├── requirements.txt
└── README.md
```

---

## 🔍 Dataset

- **Source:** [UCI Machine Learning Repository — HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- **Subjects:** 30 volunteers, 70/30 train-test split
- **Features:** 561 time- and frequency-domain features from acc/gyro signals
- **Sampling rate:** 50 Hz, sliding window of 2.56 sec (128 readings/window)

---

## ⚙️ Pipeline

```
Raw sensor data
    ↓
Exploratory Data Analysis (distribution, signal plots)
    ↓
Preprocessing (LabelEncoder + StandardScaler, stratified split)
    ↓
Model Training (LR, KNN, DT, RF, SVM) + 5-fold Cross-Validation
    ↓
GridSearchCV Hyperparameter Tuning (Random Forest)
    ↓
Evaluation (Accuracy, F1, Precision, Recall, Confusion Matrix)
    ↓
PCA Visualization + Feature Importance
    ↓
Export artifacts (pkl) + Streamlit App
```

---

## 📊 Results

| Model               | Test Accuracy | F1 (macro) | CV Mean |
|---------------------|--------------|------------|---------|
| Random Forest (Tuned) | **~96%**   | **~0.96**  | ~0.95   |
| SVM (RBF)           | ~95%         | ~0.95      | ~0.94   |
| Logistic Regression | ~94%         | ~0.94      | ~0.93   |
| Decision Tree       | ~87%         | ~0.87      | ~0.86   |
| KNN (k=5)           | ~91%         | ~0.91      | ~0.90   |

> Results may vary slightly due to random state and CV folds.

---

## 🚀 Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (needs train.csv in working directory)
python har_train.py

# Run the Streamlit app
streamlit run app.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit
```

---

## 🧠 Key Learnings

- **Sensor signals** for static activities (LAYING, SITTING, STANDING) are highly separable in PCA space; dynamic activities (WALKING variants) are more overlapping
- **Random Forest** outperforms other models thanks to ensemble averaging over 561 noisy features
- **Cross-validation** is critical — a single train-test split can be misleading; CV std tells the true story
- **Feature importance** shows that mean body acceleration features dominate — frequency domain features add marginal value

---

## 📈 Future Work

- [ ] LSTM/Transformer on raw time-series (pre-windowing)
- [ ] SHAP explainability for per-prediction feature attribution
- [ ] Real-time inference via phone sensor API
- [ ] Docker deployment

---
