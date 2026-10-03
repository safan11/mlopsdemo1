import joblib

model = joblib.load("model.pkl")

# Try a new student who studied 7 hours.
study_hours = [[2]]
prediction = model.predict(study_hours)

print("Prediction:", prediction)
print("PASS" if prediction[0] == 1 else "FAIL")
