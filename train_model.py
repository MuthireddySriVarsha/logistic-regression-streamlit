import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import sklearn
print(sklearn.__version__)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\muthi\Downloads\Logistic Regression\Logistic Regression\Titanic_train.csv")   # Titanic dataset

X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

# -----------------------------
# Preprocessing
# -----------------------------
num_features = ["Age", "Fare", "SibSp", "Parch"]
cat_features = ["Sex", "Embarked", "Pclass"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# -----------------------------
# Model pipeline
# -----------------------------
model = Pipeline([
    ("preprocessing", preprocessor),
    ("lr", LogisticRegression(max_iter=1000))
])

# -----------------------------
# Train
# -----------------------------
model.fit(X, y)

# -----------------------------
# Save model
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl created successfully")
