# Logistic-Regression
 Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target variable (species)
# Convert to binary classification (for example, classifying if species is not 0)
y_binary = (y != 0).astype(int)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200)
# Fit the model to the training data
model.fit(X_train, y_train)
# Predict species for the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
# Print classification report and confusion matrix for detailed evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
