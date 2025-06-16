# 🍽️ Restaurant Revenue Prediction with Machine Learning

## 📁 Project Overview

This project focuses on predicting restaurant revenue using a variety of regression models. The workflow includes data preprocessing, handling missing values, encoding, normalization, feature selection, and comprehensive model comparison. Multiple models were trained and evaluated, and hyperparameter tuning was conducted using **RandomizedSearchCV** for optimization.

---

## 🎯 Objectives

- Load and explore structured restaurant revenue data
- Preprocess and clean data (handle nulls, encode categories, normalize)
- Perform feature selection using Random Forest importance
- Train and evaluate regression models (RF, SVR, MLP, SGD, Decision Tree)
- Perform hyperparameter tuning with RandomizedSearchCV
- Compare performance using RMSE

---

## 📊 Dataset Details

- **Source**: Kaggle (Restaurant Revenue Prediction dataset)
- **Train File**: `train1.csv`
- **Test File**: `test1.csv`
- **Target Variable**: `revenue` (continuous)
- **Features**: A mix of numerical and categorical (City, City Group, Type)
- **Size**: ~137 columns and several thousand rows

---

## 🧪 Models Used

| Model                  | Notes                          |
|------------------------|---------------------------------|
| Random Forest Regressor| Tree-based ensemble model       |
| Decision Tree Regressor| Simple baseline model           |
| SVR (Support Vector)   | Kernel-based regression         |
| MLP Regressor          | Feed-forward neural network     |
| SGD Regressor          | Stochastic gradient descent     |

---

## 🧠 Workflow Summary

1. **Data Loading & EDA**
   - Loaded `train1.csv` and `test1.csv`
   - Explored columns, missing values, revenue distribution

2. **Preprocessing**
   - Handled null values with `SimpleImputer`
   - Dropped irrelevant columns: `Id`, `Open Date`
   - Encoded categorical variables with `OneHotEncoder`
   - Scaled numeric features using `StandardScaler`

3. **Feature Selection**
   - Used `SelectFromModel` on a Random Forest Regressor to pick top features

4. **Train/Test Split**
   - Split data using `train_test_split` (80/20)

5. **Model Training & Evaluation**
   - Trained five models and evaluated using **Root Mean Squared Error (RMSE)**

6. **Hyperparameter Tuning**
   - Performed `RandomizedSearchCV` on all five models to identify optimal parameters

---

## 📈 Key Results

- All models were evaluated based on **RMSE**
- Feature selection significantly improved training efficiency
- Random Forest and SVR delivered the best RMSE before tuning
- Hyperparameter tuning improved MLP and SGD significantly

---

## 🛠️ Tools & Libraries

- **Python 3.x**
- **Pandas, NumPy** for data handling
- **Scikit-learn** for modeling and evaluation
- **Matplotlib, Seaborn** for visualization
- **RandomizedSearchCV** for tuning
- **OneHotEncoder, ColumnTransformer, SelectFromModel**

---

## 👤 About Me

I’m **Balbir Singh**, a data and business analyst passionate about applying machine learning to real-world problems
---

## 📌 Acknowledgment

Thanks to **Kaggle** and the dataset providers for enabling hands-on learning in regression modeling and revenue prediction.
