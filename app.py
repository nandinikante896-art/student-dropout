from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    gender = request.form["gender"]
    admission_grade = float(request.form["admission_grade"])
    attendance = float(request.form["attendance"])
    study_hours = float(request.form["study_hours"])
    scholarship = request.form["scholarship"]
    tuition = request.form["tuition"]
    parents = request.form["parents"]
    income = float(request.form["income"])
    exam_score = float(request.form["exam_score"])

    # Encode categorical
    gender = encoders["gender"].transform([gender])[0]
    scholarship = encoders["scholarship"].transform([scholarship])[0]
    tuition = encoders["tuition_fees_paid"].transform([tuition])[0]
    parents = encoders["parents_education"].transform([parents])[0]

    input_features = np.array([[age, gender, admission_grade, attendance,
                                study_hours, scholarship, tuition,
                                parents, income, exam_score]])

    prediction = model.predict(input_features)[0]
    prob = prediction  # Since it's regression, use the score directly

    if prediction >= 0.5:
        result = "⚠️ Student likely to Dropout"
    else:
        result = "✅ Student likely to Continue"

    probability = round(prob*100,2)

    return render_template("result.html", prediction=result, prob=probability)

if __name__ == "__main__":
    app.run(debug=True)