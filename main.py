import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Load training data
df = pd.read_csv("data/train.csv")

# Drop unnecessary columns
df = df.drop(["Name", "SibSp", "Parch", "Embarked", "Ticket", "PassengerId", "Cabin"], axis=1)

# Handle missing values
df = df.dropna()

# Convert categorical 'Sex' column to numerical (0: Female, 1: Male)
df["Sex"] = df["Sex"].astype("category").cat.codes

# Correlation heatmap (optional)
# sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
# plt.show()

# Split data into features (X) and target (y)
x = df.drop(["Survived"], axis=1)
y = df["Survived"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(x_train, y_train)

# Load test data
df2 = pd.read_csv("data/test.csv")

# Drop unnecessary columns
df2 = df2.drop(["Name", "SibSp", "Parch", "Embarked", "Ticket", "PassengerId", "Cabin"], axis=1)

# Handle missing values (fill with mean instead of dropping)
df2.fillna(df2["Age"].mean(), inplace=True)

# Convert categorical 'Sex' column to numerical
df2["Sex"] = df2["Sex"].astype("category").cat.codes

# Evaluate model accuracy on test set
print("Test Accuracy:", model.score(x_test, y_test))

# Make predictions on the test dataset
predictions = model.predict(df2)

# Load gender_submission.csv to compare predictions
df3 = pd.read_csv("data/gender_submission.csv")

# Ensure both datasets have the same length before comparison
n = min(len(df3), len(predictions))

# Compare predicted vs actual survival
correct_predictions = sum(predictions[:n] == df3["Survived"][:n])
print(f"Correct Predictions: {correct_predictions}/{n}")
