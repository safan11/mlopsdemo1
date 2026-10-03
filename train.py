from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# Study hours for each student
X = [[1], [2], [3], [4], [5], [6]]

# 0 = Fail, 1 = Pass
y = [0, 0, 0, 1, 1, 1]

mlflow.set_experiment("Student_Pass_Fail")

with mlflow.start_run():
    max_depth = 2
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    mlflow.log_param("model_type", "DecisionTree")
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # Keep a local artifact for predict.py.
    joblib.dump(model, "model.pkl")

    print("Accuracy:", accuracy)
    print("Model trained and saved to model.pkl")
    print("Model and run logged to MLflow!")
