import sys
from pathlib import Path
from io import BytesIO
import tempfile
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import scipy.io


# ----------------------------------
# 1. PROJECT PATHS
# ----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import extract_features_from_mat


MODEL_PATH = PROJECT_ROOT / "results" / "models" / "random_forest_model.pkl"

ENCODER_PATH = PROJECT_ROOT / "results" / "models" / "label_encoder.pkl"


# ----------------------------------
# 2. LOAD TRAINED MODEL
# ----------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    encoder = joblib.load(ENCODER_PATH)

    explainer = shap.TreeExplainer(model)

    return model, encoder, explainer


rf_model, label_encoder, explainer = load_model()


# ----------------------------------
# 3. STREAMLIT PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Industrial Fault Diagnosis",
    page_icon="⚙️",
    layout="wide"
)


# ----------------------------------
# 4. APPLICATION TITLE
# ----------------------------------

st.title("⚙️ Explainable Industrial Fault Diagnosis")

st.write(
    "An intelligent bearing fault diagnosis system "
    "using Random Forest and SHAP Explainable AI."
)

st.divider()


# ----------------------------------
# 5. FILE UPLOAD
# ----------------------------------

st.subheader("Upload Vibration Signal")

uploaded_file = st.file_uploader(
    "Upload a MATLAB (.mat) vibration file",
    type=["mat"]
)


# ----------------------------------
# 6. PROCESS UPLOADED FILE
# ----------------------------------

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    try:

                # Extract features from uploaded file
        with tempfile.TemporaryDirectory() as temp_dir:

            # Preserve the original uploaded filename
            temp_path = Path(temp_dir) / Path(uploaded_file.name).name

            # Save uploaded file temporarily
            temp_path.write_bytes(uploaded_file.getvalue())

            # Extract features using the file path
            features_df = extract_features_from_mat(temp_path)

        

        st.subheader("Extracted Features")

        st.write(
            f"Total vibration segments: {len(features_df)}"
        )

        st.dataframe(features_df.head())

        # ----------------------------------
        # 7. MODEL PREDICTION
        # ----------------------------------

        probabilities = rf_model.predict_proba(features_df)

        # Average probabilities across all segments
        average_probabilities = probabilities.mean(axis=0)

        predicted_class = np.argmax(average_probabilities)

        predicted_label = label_encoder.inverse_transform(
            [predicted_class]
        )[0]

        confidence = average_probabilities[predicted_class] * 100

        st.divider()

        st.subheader("Bearing Fault Diagnosis")

        st.success(f"Predicted Condition: {predicted_label}")

        st.metric(
            "Average Model Probability",
            f"{confidence:.2f}%"
        )

        # ----------------------------------
        # 8. CLASS PROBABILITIES
        # ----------------------------------

        st.subheader("Prediction Probabilities")

        probability_df = pd.DataFrame({
            "Bearing Condition": label_encoder.classes_,
            "Probability (%)": average_probabilities * 100
        })

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            probability_df.set_index("Bearing Condition")
        )

        # ----------------------------------
        # 9. SHAP EXPLANATION
        # ----------------------------------

        st.divider()

        st.subheader("Explainable AI - SHAP Analysis")

        # Select the segment with the highest probability
        # for the final predicted class
        representative_idx = np.argmax(
            probabilities[:, predicted_class]
        )

        sample = features_df.iloc[[representative_idx]]

        # Calculate SHAP values
        shap_values = explainer.shap_values(sample)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):

            class_shap_values = shap_values[predicted_class][0]

        else:

            shap_values = np.asarray(shap_values)

            if shap_values.ndim == 3:

                class_shap_values = shap_values[
                    0, :, predicted_class
                ]

            else:

                class_shap_values = shap_values[0]

        base_values = np.asarray(explainer.expected_value)

        if base_values.ndim == 0:

            base_value = float(base_values)

        else:

            base_value = float(base_values[predicted_class])

        explanation = shap.Explanation(
            values=class_shap_values,
            base_values=base_value,
            data=sample.iloc[0].values,
            feature_names=sample.columns.tolist()
        )

        st.write(
            "The waterfall plot explains which features "
            "influenced the selected segment's prediction."
        )

        shap.plots.waterfall(
            explanation,
            max_display=10,
            show=False
        )

        fig = plt.gcf()

        st.pyplot(fig)

        plt.close(fig)

        st.info(
            "SHAP explains the selected representative segment, "
            "while the final diagnosis uses average probabilities "
            "across all segments."
        )

    except Exception as e:

        st.error(f"An error occurred: {e}")