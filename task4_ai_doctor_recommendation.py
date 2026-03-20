import pandas as pd
import numpy as np
import warnings
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ── Load API Key ───────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found! Check your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)
print('Gemini configured successfully!')

# ── Load & Prepare Data ────────────────────────────────────
df = pd.read_csv('data/healthcare_dataset.csv')
df_model = df.copy()

df_model['Date of Admission']   = pd.to_datetime(df_model['Date of Admission'])
df_model['Discharge Date']      = pd.to_datetime(df_model['Discharge Date'])
df_model['Length of Stay']      = (df_model['Discharge Date'] - df_model['Date of Admission']).dt.days
df_model['Admission Month']     = df_model['Date of Admission'].dt.month
df_model['Admission DayOfWeek'] = df_model['Date of Admission'].dt.dayofweek
df_model.drop(columns=['Name','Doctor','Hospital','Date of Admission','Discharge Date'], inplace=True)

cat_cols = ['Gender','Blood Type','Medical Condition','Insurance Provider','Admission Type','Medication']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le
    print(f'{col} classes: {le.classes_}')   # ← so you can see valid values

le_target = LabelEncoder()
df_model['Test Results'] = le_target.fit_transform(df_model['Test Results'])

X = df_model.drop(columns=['Test Results'])
y = df_model['Test Results']
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)
print(f'\nModel trained | Target classes: {le_target.classes_}')

# ── Prediction Function ────────────────────────────────────
def predict_test_result(patient_dict):
    row = pd.DataFrame([patient_dict])
    for col in cat_cols:
        row[col] = label_encoders[col].transform(row[col])
    row = row[X.columns]
    pred  = rf.predict(row)[0]
    label = le_target.inverse_transform([pred])[0]
    conf  = max(rf.predict_proba(row)[0]) * 100
    return label, conf

# ── Gemini Recommendation Function ────────────────────────
def generate_recommendation(patient_info, predicted_result, confidence):
    prompt = f"""
You are an experienced clinical doctor. Based on the patient details below,
write a professional and concise medical recommendation.

Patient Information:
- Name: {patient_info['name']}
- Age: {patient_info['Age']} years
- Gender: {patient_info['Gender']}
- Blood Type: {patient_info['Blood Type']}
- Medical Condition: {patient_info['Medical Condition']}
- Current Medication: {patient_info['Medication']}
- Admission Type: {patient_info['Admission Type']}
- Billing Amount: ${patient_info['Billing Amount']:,.2f}

AI Predicted Test Result: {predicted_result} (Confidence: {confidence:.1f}%)

Please provide:
1. Summary — Plain-language interpretation of the test result
2. Clinical Recommendation — Advised actions (tests, medications, lifestyle)
3. Health Advice — 3 specific tips relevant to the condition and age
4. Follow-up — When and how often to return for review
5. Risk Level — Low / Moderate / High with a brief explanation

Tone: professional yet warm. Format with bold headers. 200-300 words.
"""
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    return response.text

# ── Sample Patient ─────────────────────────────────────────
# NOTE: All values must exactly match what's in the dataset
# Run this file once to see the valid classes printed above,
# then fill in the sample patient accordingly.

sample_patient = {
    'name'             : 'Rajesh Kumar',
    'Age'              : 58,
    'Gender'           : 'Male',
    'Blood Type'       : 'A+',
    'Medical Condition': 'Diabetes',
    'Insurance Provider': 'Aetna',
    'Billing Amount'   : 28500.75,
    'Room Number'      : 312,
    'Admission Type'   : 'Elective',
    'Medication'       : 'Aspirin',     # ← use a value printed in Medication classes above
    'Length of Stay'   : 7,
    'Admission Month'  : 6,
    'Admission DayOfWeek': 2
}

features_only = {k: v for k, v in sample_patient.items() if k != 'name'}
predicted, confidence = predict_test_result(features_only)

print(f'\nPatient         : {sample_patient["name"]}')
print(f'Predicted Result: {predicted}')
print(f'Confidence      : {confidence:.1f}%')
print('\n' + '='*65)
print(generate_recommendation(sample_patient, predicted, confidence))
print('='*65)