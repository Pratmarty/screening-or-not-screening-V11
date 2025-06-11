
import streamlit as st
import joblib
import numpy as np
import os

# Nouveau nom de modèle
MODEL_PATH = "rf_model_compatible.joblib"
if not os.path.exists(MODEL_PATH):
    st.error("Modèle non trouvé. Assurez-vous que 'rf_model_compatible.joblib' est dans le même dossier que app.py.")
    st.stop()

# Chargement du modèle
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Risque de Blessure V9.0", layout="wide")
st.title("🧠 Prédiction du Risque de Blessure - Version 9.0")

st.markdown("Remplis le formulaire ci-dessous pour estimer ton risque de blessure.")

with st.form("formulaire"):
    col1, col2, col3 = st.columns(3)

    with col1:
        dorsiflexion = st.slider("Amplitude dorsiflexion (en °)", 0, 40, 20)
        adducteur_strength = st.slider("Force adducteurs (kg)", 0, 100, 50)
        vertical_jump = st.slider("Saut vertical (cm)", 10, 100, 50)

    with col2:
        sprint_time = st.number_input("Temps sprint 10m (s)", 1.0, 10.0, 2.5)
        squat_1RM = st.slider("Squat 1RM (kg)", 0, 200, 100)
        charge_var = st.slider("Variation charge entraînement (%)", 0, 100, 20)

    with col3:
        fatigue = st.slider("Fatigue (1 à 5)", 1, 5, 3)
        sommeil = st.slider("Qualité du sommeil (1 à 5)", 1, 5, 3)
        historique_blessure = st.selectbox("Blessure antérieure ?", ["Non", "Oui"])

    submit = st.form_submit_button("Analyser")

if submit:
    try:
        blessure_bin = 1 if historique_blessure == "Oui" else 0

        features = np.array([
            dorsiflexion,
            adducteur_strength,
            sprint_time,
            squat_1RM,
            vertical_jump,
            charge_var,
            fatigue,
            sommeil,
            blessure_bin
        ]).reshape(1, -1)

        prediction = model.predict(features)[0]

        niveau = ["🟢 Faible", "🟠 Modéré", "🔴 Élevé"]
        st.subheader("Résultat de l'analyse")
        st.markdown(f"### Risque estimé : {niveau[prediction]}")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la prédiction : {e}")
