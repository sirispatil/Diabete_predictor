# model.py

# Import libraries
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import joblib

#  Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Lasso model
model = Lasso(alpha=1.0)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print("Model trained successfully!")
print("R2 Score:", round(score, 3))


joblib.dump(model, "best_model.pkl")

print("Model saved as best_model.pkl")