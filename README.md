
# 🛣️ Road Accident Severity Prediction Model

> Predicting the severity of road accidents using machine learning, one model at a time.

---

## 📌 Project Overview

This project aims to predict the **severity of traffic accidents** using machine learning techniques.  
The dataset is preprocessed, class-balanced using **SMOTE**, and passed through various models like **Random Forest**, **Decision Tree**, and **Logistic Regression**.  

The goal: To build a robust, interpretable pipeline that can classify accident severity with strong performance — and provide insights into what really matters on the road.

---

## 🧠 Core Objectives

- Clean and analyze real-world accident data
- Encode categorical features numerically
- Handle missing values with appropriate imputation
- Use **SMOTE** to balance class distribution
- Train multiple classification models
- Evaluate and compare model performances
- Visualize results and feature importances
- Save the best model for future use

---

## 📂 File Structure

| File | Description |
|------|-------------|
| `data_analysis.py` | Performs exploratory data analysis (EDA), basic statistics, null value checks. |
| `label_encoding.py` | Encodes categorical variables using label encoding. |
| `mod_imputation.py` | Imputes missing values based on strategies like mode/mean. |
| `smote.py` | Applies SMOTE to balance imbalanced classes in the dataset. |
| `rfc.py` | Trains and evaluates a **Random Forest** model. Also visualizes results and saves the model. |
| `rfc_test.py` | Loads the trained model and runs predictions on test data. |
| `decisionTree.py` | Trains a **Decision Tree Classifier** and evaluates its performance. |
| `logisticRegression.py` | Trains a **Logistic Regression** model for classification. |
| `guncel_dosya.xlsx` | Main dataset used in the project, containing accident records (already SMOTE-applied). |

---

## 📊 Machine Learning Pipeline

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical features
   - Normalization (optional)

2. **Class Balancing**
   - SMOTE (Synthetic Minority Oversampling Technique)

3. **Modeling**
   - Random Forest (primary model)
   - Logistic Regression
   - Decision Tree

4. **Evaluation Metrics**
   - Accuracy
   - Classification Report (Precision, Recall, F1-score)
   - Feature Importance (for interpretability)

5. **Model Saving**
   - Saved as `.joblib` file using `joblib` for later use

---

## 📈 Example Output (Random Forest)

```
Training set size: (X, Y)
Test set size: (X, Y)
Model accuracy: 0.8732

Classification Report:
              precision    recall  f1-score   support
      ...
```

Visualizations:
- Accuracy bar plot
- Feature importance chart

---

## 🔧 Installation

Make sure you have Python 3.7+ installed. Then install the required libraries:

```bash
pip install pandas scikit-learn matplotlib joblib openpyxl
```

---

## ▶️ How to Run

```bash
# Step-by-step execution (recommended order)
python data_analysis.py
python label_encoding.py
python mod_imputation.py
python smote.py
python rfc.py
python rfc_test.py
```

Optional alternatives:
```bash
python decisionTree.py
python logisticRegression.py
```

---

## 💡 Insights and Takeaways

- **Random Forest** performs best in this case with strong accuracy and generalization.
- **Feature importance** plots reveal which variables influence accident severity most (e.g., road type, weather, vehicle type).
- **SMOTE** proves crucial in handling class imbalance, greatly improving minority class predictions.

---

## 🌟 What's Next?

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Cross-validation for more robust evaluation
- Deploy as a REST API with Flask or FastAPI
- Build a front-end to make predictions from user input

---

## 🤝 Contributions

If you have ideas for improvements, pull requests are welcome. Let’s make safer roads together, one prediction at a time.

---
