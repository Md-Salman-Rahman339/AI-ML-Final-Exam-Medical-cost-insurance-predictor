# 💰 Medical Insurance Cost Prediction

## 📌 Project Overview

This project builds a **Machine Learning regression model** to predict **medical insurance charges** based on personal and demographic features such as age, BMI, smoking habits, and region.

The model is trained using the **Insurance dataset** and optimized using **Random Forest Regressor with hyperparameter tuning**.

---

## 🎯 Objectives

* Predict medical insurance cost accurately
* Perform full data preprocessing pipeline
* Apply feature engineering techniques
* Optimize model performance using GridSearchCV
* Evaluate model using multiple metrics

---

## 📂 Dataset Information

The dataset contains **1338 rows and 7 columns**:

| Feature  | Description                     |
| -------- | ------------------------------- |
| age      | Age of the person               |
| sex      | Gender (male/female)            |
| bmi      | Body Mass Index                 |
| children | Number of children              |
| smoker   | Smoking status (yes/no)         |
| region   | Residential area                |
| charges  | Medical insurance cost (target) |

---

## ⚙️ Data Preprocessing

### ✔️ Steps Performed:

* Checked missing values (none found)
* Encoded categorical variables
* Outlier detection using boxplots
* Outlier handling using IQR method

---

## 🧠 Feature Engineering

New features created:

* `bmi_category` → Categorized BMI
* `age_group` → Age segmentation
* `family_size` → children + 1

---

## 📊 Exploratory Data Analysis

* Correlation heatmap used to analyze relationships
* Strong correlation found between **smoker and charges**
* Moderate correlation with **age and BMI**

---

## 🔀 Train-Test Split

* Training set: 80%
* Testing set: 20%
* `random_state = 42` for reproducibility

---

## 🏗️ Model Pipeline

Pipeline includes:

* **StandardScaler** for numerical features
* **OneHotEncoder** for categorical features
* **RandomForestRegressor** as the model

---

## 🤖 Model Selection

Random Forest Regressor was chosen because:

* Handles both numerical & categorical data
* Captures non-linear relationships
* Reduces overfitting via ensemble learning
* Performs well on tabular datasets

---

## 📈 Model Performance

### 🔹 Training Performance:

* R² Score: **0.9749**
* RMSE: **1900.98**
* MAE: **1035.86**

### 🔹 Cross Validation:

* Mean R²: **0.8281**
* Std Dev: **0.0401**

---

## 🔧 Hyperparameter Tuning

Used **GridSearchCV** with:

* n_estimators: [100, 200]
* max_depth: [None, 10, 20]
* min_samples_split: [2, 5]

### ✅ Best Parameters:

* max_depth = 10
* min_samples_split = 5
* n_estimators = 200

### ⭐ Best CV Score:

* **0.8338**

---

## 📊 Final Model Evaluation

* Test R² Score: **0.8687**
* Test RMSE: **4515.38**
* Test MAE: **2519.66**

---

## 💾 Model Saving

```python
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

with open('features_names.pkl',"wb") as f:
    pickle.dump(X.columns, f)
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2️⃣ Run Notebook / Script

```bash
jupyter notebook
```

---

## 🖥️ Future Improvements
* Add real-time user interface
* Use advanced models (XGBoost, LightGBM)
* Improve feature engineering

---

## 📌 Conclusion

The model successfully predicts insurance charges with **high accuracy (R² ≈ 0.87)**. Smoking status plays a major role in determining insurance costs, followed by age and BMI.

---
Deployment link: https://huggingface.co/spaces/salman339/Medical-Cost-insurance-Predictor
## 👨‍💻 Author

**Md Salman Rahman**
BSc in CSE | Machine Learning Enthusiast

---
