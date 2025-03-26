# TitanicSurvivors
<h1 align="center">🚢 Titanic Survival Prediction - Logistic Regression</h1>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/6e/Titanic_PCS.jpg" width="500" alt="Titanic">
</p>

## 📌 Project Overview
This project predicts whether a passenger survived the **Titanic disaster** using **Logistic Regression**.  
It involves **feature engineering, data preprocessing, training a model, and evaluating performance**.

---

## 🛠️ Technologies Used
- Python 🐍
- Pandas 📊
- Scikit-Learn 🤖
- Seaborn 🎨
- Matplotlib 📈

---

## 📂 Dataset Used
The dataset is taken from **Kaggle’s Titanic Competition**, which contains:
- `train.csv` - Training data with survival labels.
- `test.csv` - Test data without labels.
- `gender_submission.csv` - Sample submission file.

---

## 🔍 Feature Engineering & Data Preprocessing

### 🔹 1. **Data Cleaning**
- **Dropped unnecessary columns**: `"Name", "SibSp", "Parch", "Embarked", "Ticket", "PassengerId", "Cabin"`
- **Handled missing values**:
  - Filled missing **numerical values** with their column **mean**.
  - Converted **categorical values** (`Sex`) into numerical form (`0 = Female`, `1 = Male`).

### 🔹 2. **Exploratory Data Analysis (EDA)**
- Used **correlation heatmaps** to identify important features.
- Checked survival rates based on **gender, age, and class**.

### 🔹 3. **Model Training**
- **Train-Test Split**: 80% training, 20% testing.
- **Trained a Logistic Regression Model** on `x_train, y_train`.
- Evaluated performance using `accuracy_score()`.

### 🔹 4. **Prediction & Evaluation**
- **Made predictions** on `test.csv`.
- Compared predictions with `gender_submission.csv` for accuracy.

---

## 📊 Model Performance
- **Training Accuracy**: ✅ `model.score(x_train, y_train)`
- **Test Accuracy**: ✅ `model.score(x_test, y_test)`
- **Final Predictions compared against ground truth.**

---

## 🏆 Results
The model successfully predicts **survival chances** based on given passenger details.  
Future improvements could involve:
- Using **Random Forests** or **XGBoost** for better accuracy.
- Handling missing values in a more sophisticated way.

---

## 🚀 How to Run the Code?
1. Clone the repository:  
   ```bash
   git clone https://github.com/tanishkdubey/TitanicSurvivors.git

