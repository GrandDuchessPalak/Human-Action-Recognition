# 🏃 Human Activity Recognition (HAR) — Sensor-Based ML Classification

> Classifying six daily activities from smartphone accelerometer & gyroscope data  
> using five machine learning models. Best result: **Random Forest ~99% accuracy**.

---

## 📌 Overview

This project builds and evaluates multiple supervised ML classifiers on the  
[UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones),  
which contains 10,299 samples of 561 pre-computed inertial sensor features  
collected from 30 subjects wearing a waist-mounted smartphone.

**Activities classified:**

| Label | Activity |
|-------|----------|
| 0 | LAYING |
| 1 | SITTING |
| 2 | STANDING |
| 3 | WALKING |
| 4 | WALKING_DOWNSTAIRS |
| 5 | WALKING_UPSTAIRS |

---

## 🧠 Models & Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Random Forest | ~99% | ~99% |
| SVM (RBF) | ~99% | ~99% |
| Logistic Regression | ~96% | ~96% |
| Decision Tree | ~87% | ~87% |
| KNN (best k) | ~91% | ~91% |

---

## 📁 Project Structure

```
har-activity-recognition/
├── HAR_Activity_Recognition.ipynb   # Main notebook
├── README.md                        # This file
├── har_model.pkl                    # Saved Random Forest model
├── scaler.pkl                       # Fitted StandardScaler
├── label_encoder.pkl                # LabelEncoder (int → activity)
└── feature_names.pkl                # 561 feature names
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/har-activity-recognition.git
cd har-activity-recognition
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 3. Download the dataset
Download `train.csv.zip` from the  
[UCI HAR Dataset page](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)  
and place it in the project root.

### 4. Run the notebook
```bash
jupyter notebook HAR_Activity_Recognition.ipynb
```

---

## 🔍 Key Findings

- **Random Forest** and **SVM (RBF)** both achieve ~99% accuracy, far outperforming simpler models.
- **PCA visualization** reveals that static activities (laying, sitting, standing) and dynamic activities (walking variants) naturally separate into two super-clusters — explaining why even linear models perform well.
- **Top feature importances** are frequency-domain magnitude statistics (FFT-derived), confirming that spectral features are the most discriminative signals.
- The dataset is well-balanced (~16–17% per class), so no resampling was needed.

---

## 🛠️ Tech Stack

- **Python 3.10**
- **scikit-learn** — model training, evaluation, preprocessing
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization
- **joblib** — model serialization

---

## 📈 Possible Extensions

- Hyperparameter tuning with `GridSearchCV` for RF / SVM
- Try **XGBoost** or **LightGBM** for potential gains
- Deploy as a **FastAPI REST endpoint** using saved `.pkl` artefacts
- Apply on **raw sensor windows** using CNN or LSTM (time-series deep learning)

---

## 📄 License

MIT License — free to use and adapt with attribution.

---

## 🙋 Author

**Palak**  
[LinkedIn](https://www.linkedin.com/in/palak-arora-3a1296325?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) · [GitHub](https://github.com/GrandDuchessPalak)

---

*Dataset: [UCI Machine Learning Repository — HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)*
