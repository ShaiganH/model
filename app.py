from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models
rf_diagnosis = joblib.load("rf_diagnosis_model.pkl")
rf_severity = joblib.load("rf_severity_model.pkl")

# Load encoders
enc_gender = joblib.load("Gender_encoder.pkl")
enc_sym1 = joblib.load("Symptom_1_encoder.pkl")
enc_sym2 = joblib.load("Symptom_2_encoder.pkl")
enc_sym3 = joblib.load("Symptom_3_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        features = np.array([[
            data['Age'],
            enc_gender.transform([data['Gender']])[0],
            enc_sym1.transform([data['Symptom_1']])[0],
            enc_sym2.transform([data['Symptom_2']])[0],
            enc_sym3.transform([data['Symptom_3']])[0],
            data['Heart_Rate_bpm'],
            data['Body_Temperature_C'],
            data['Oxygen_Saturation_%'],
            data['Systolic'],
            data['Diastolic']
        ]])

        diagnosis = rf_diagnosis.predict(features)[0]
        severity = rf_severity.predict(features)[0]

        return jsonify({
            "Diagnosis": diagnosis,
            "Severity": severity
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)