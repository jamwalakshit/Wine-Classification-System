import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/Wine.csv")

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("Customer_Segment", axis=1)
y = df["Customer_Segment"]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# -----------------------------
# Test Accuracy
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "model/wine_model.pkl")

print("Model Saved Successfully!")