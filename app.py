import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Healthcare AI Decision Support", page_icon="🏥", layout="centered")

# Sidebar navigation (clean names without accidental spaces)
st.sidebar.title("Select ML Project")
project = st.sidebar.selectbox(
    "Choose Project:",
    ["Drug Classification", "Chronic Disease Progression", "Diabetes Risk Prediction"]
)

# ==========================================
# 1. DRUG CLASSIFICATION
# ==========================================
if project == "Drug Classification":
    st.title("💊 Clinical Drug Recommendation System")
    st.write("Prescribe medications based on electrolytes and patient vitals.")
    
    @st.cache_resource
    def load_drug_model():
        df = pd.read_csv('drug200.csv')
        encoders = {}
        for col in ['Sex', 'BP', 'Cholesterol']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        target_le = LabelEncoder()
        df['Drug_Target'] = target_le.fit_transform(df['Drug'])
        encoders['Drug'] = target_le
        X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
        y = df['Drug_Target']
        model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
        model.fit(X, y)
        return model, encoders

    try:
        model, encoders = load_drug_model()
        age = st.slider("Patient Age", 15, 80, 35)
        na_to_k = st.slider("Na_to_K Ratio", 5.0, 40.0, 10.0, 0.1)
        col1, col2, col3 = st.columns(3)
        with col1:
            sex = st.selectbox("Sex", ['F', 'M'])
        with col2:
            bp = st.selectbox("Blood Pressure", ['HIGH', 'NORMAL', 'LOW'])
        with col3:
            chol = st.selectbox("Cholesterol", ['HIGH', 'NORMAL'])

        if st.button("Recommend Drug", type="primary"):
            input_df = pd.DataFrame([[
                age,
                encoders['Sex'].transform([sex])[0],
                encoders['BP'].transform([bp])[0],
                encoders['Cholesterol'].transform([chol])[0],
                na_to_k
            ]], columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
            
            pred = model.predict(input_df)[0]
            drug = encoders['Drug'].inverse_transform([pred])[0]
            conf = model.predict_proba(input_df)[0][pred] * 100
            st.success(f"### Prescribed Drug: **{drug}**")
            st.metric("Confidence Score", f"{conf:.1f}%")
    except FileNotFoundError:
        st.error("Error: `drug200.csv` file not found!")

# ==========================================
# 2. CHRONIC DISEASE PROGRESSION
# ==========================================
elif project == "Chronic Disease Progression":
    st.title("🏥 Chronic Disease Diagnosis System")
    st.write("Diagnose Alzheimer's, Diabetes, or Parkinson's using clinical biomarker panels.")
    
    @st.cache_resource
    def load_chronic_model():
        df = pd.read_csv('chronic_disease_progression.csv')
        df['MedicalHistory'] = df['MedicalHistory'].fillna('None')
        df['AlcoholUse'] = df['AlcoholUse'].fillna('None')
        df = df.drop(columns=['PatientID', 'Date'])
        encoders = {}
        for col in df.select_dtypes(include=['object']).columns.drop('Disease'):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        target_le = LabelEncoder()
        df['Disease_Target'] = target_le.fit_transform(df['Disease'])
        encoders['Disease'] = target_le
        X = df.drop(columns=['Disease', 'Disease_Target'])
        y = df['Disease_Target']
        model = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42)
        model.fit(X, y)
        return model, encoders, list(X.columns)

    try:
        model, encoders, feature_names = load_chronic_model()
        biomarker = st.slider("Biomarker Score", 0.0, 60.0, 20.0, 0.1)
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 40, 90, 65)
            bmi = st.number_input("BMI", 12.0, 45.0, 26.5, 0.1)
            sbp = st.slider("Systolic BP", 110, 160, 135)
            cog = st.slider("Cognitive Score", 10, 30, 20)
        with col2:
            stage = st.selectbox("Stage", ['Early', 'Mid', 'Late'])
            lifestyle = st.selectbox("Lifestyle", ['Active', 'Sedentary', 'Non-Smoker', 'Smoker'])
            history = st.selectbox("Medical History", ['None', 'Heart Disease', 'Hypertension', 'Stroke', 'Asthma'])

        if st.button("Diagnose Condition", type="primary"):
            row = {
                'Age': age, 'Gender': encoders['Gender'].transform(['Male'])[0],
                'MedicalHistory': encoders['MedicalHistory'].transform([history])[0],
                'Lifestyle': encoders['Lifestyle'].transform([lifestyle])[0],
                'BiomarkerScore': biomarker, 'MedicationDose': 1.25, 'HeartRate': 80,
                'BloodPressure_Systolic': sbp, 'BloodPressure_Diastolic': 85, 'Cholesterol': 200,
                'BMI': bmi, 'SleepHours': 6.5, 'StepsPerDay': 6500,
                'MedicationAdherence': encoders['MedicationAdherence'].transform(['High'])[0],
                'StressLevel': 5, 'CognitiveScore': cog, 'MoodScore': 5,
                'Stage': encoders['Stage'].transform([stage])[0], 'Smoker': 0,
                'AlcoholUse': encoders['AlcoholUse'].transform(['None'])[0],
                'SupportSystem': encoders['SupportSystem'].transform(['Moderate'])[0],
                'HasCaregiver': 1, 'EmploymentStatus': encoders['EmploymentStatus'].transform(['Retired'])[0]
            }
            input_df = pd.DataFrame([row])[feature_names]
            pred = model.predict(input_df)[0]
            condition = encoders['Disease'].inverse_transform([pred])[0]
            conf = model.predict_proba(input_df)[0][pred] * 100
            st.success(f"### Diagnosis: **{condition}**")
            st.metric("Confidence Score", f"{conf:.1f}%")
    except FileNotFoundError:
        st.error("Error: `chronic_disease_progression.csv` file not found!")

# ==========================================
# 3. DIABETES RISK PREDICTION
# ==========================================
elif project == "Diabetes Risk Prediction":
    st.title("🩺 Diabetes Diagnostic Prediction System")
    st.write("Evaluate patient diabetes risk probability.")
    
    @st.cache_resource
    def load_diabetes_model():
        df = pd.read_csv('Diabetes_prediction.csv')
        X = df.drop(columns=['Diagnosis'])
        y = df['Diagnosis']
        model = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
        model.fit(X, y)
        return model, list(X.columns)

    try:
        model, feature_names = load_diabetes_model()
        col1, col2 = st.columns(2)
        with col1:
            glucose = st.slider("Glucose Level (mg/dL)", 40.0, 200.0, 120.0, 0.5)
            bp = st.slider("Blood Pressure", 40.0, 130.0, 75.0, 0.5)
            bmi = st.slider("BMI", 15.0, 45.0, 28.0, 0.1)
        with col2:
            insulin = st.slider("Insulin", 0.0, 300.0, 90.0, 1.0)
            dpf = st.slider("Diabetes Pedigree Function", 0.05, 1.5, 0.45, 0.01)
            age = st.slider("Age", 15.0, 90.0, 45.0, 1.0)

        if st.button("Evaluate Diabetes Risk", type="primary"):
            input_df = pd.DataFrame([[1, glucose, bp, 23.0, insulin, bmi, dpf, age]], columns=feature_names)
            prob = model.predict_proba(input_df)[0][1]
            if prob >= 0.35 or glucose >= 140.0:
                st.error("### Result: **High Diabetes Risk Detected (Positive)**")
                st.metric("Risk Probability", f"{prob * 100:.1f}%")
            else:
                st.success("### Result: **Low Risk / Non-Diabetic (Negative)**")
                st.metric("Risk Probability", f"{prob * 100:.1f}%")
    except FileNotFoundError:
        st.error("Error: `Diabetes_prediction.csv` file not found!")