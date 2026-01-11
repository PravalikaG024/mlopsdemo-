
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Step 1: Create dataset
data = {
    "study_hours": [1,2,3,4,5,6,7,8,9,10],
    "attendance": [50,55,60,65,70,75,80,85,90,95],
    "score": [35,40,45,50,55,60,65,70,75,80]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance"]]
y = df["score"]


pipeline = Pipeline(steps=[("scaler", StandardScaler()),("model", LinearRegression())])

pipeline.fit(X, y)


joblib.dump(pipeline, "model.pkl")

print("Model trained and saved as model.pkl")


