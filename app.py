import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians
from sklearn.metrics.pairwise import haversine_distances
from opencage.geocoder import OpenCageGeocode
import shap
import matplotlib.pyplot as plt
import os

# ---------- Load Data and Model ----------
@st.cache_data
def load_data():
    filepath = os.path.join(os.path.dirname(__file__), "synthetic_pmjay_dataset.csv")
    return pd.read_csv(filepath)

@st.cache_resource
def load_model():
    filepath = os.path.join(os.path.dirname(__file__), "hospital_xgb_pipeline.pkl")
    return joblib.load(filepath)

df = load_data()
model = load_model()
preprocessor = model.named_steps['preprocessor']
regressor = model.named_steps['xgb']

# ---------- Recommender Function ----------
def recommend_hospitals(user_lat, user_lon, user_state, disease_query, top_n=5):
    df_filtered = df[df['disease_query'].str.lower() == disease_query.lower()].copy()
    if df_filtered.empty:
        return None, None, None, None

    user_coords = np.array([[radians(user_lat), radians(user_lon)]])
    hosp_coords = np.radians(df_filtered[['latitude', 'longitude']].values)
    distances = haversine_distances(user_coords, hosp_coords) * 6371
    df_filtered['distance_km'] = distances.flatten()

    X_raw = df_filtered[['rating', 'distance_km', 'match_score']]
    X_transformed = preprocessor.transform(X_raw)
    df_filtered['predicted_score'] = regressor.predict(X_transformed)

    df_filtered['boost'] = df_filtered['state'].apply(lambda s: 1.1 if user_state.lower() in s.lower() else 1.0)
    df_filtered['boosted_score'] = df_filtered['predicted_score'] * df_filtered['boost']

    explainer = shap.Explainer(regressor, feature_names=['rating', 'distance_km', 'match_score'])
    shap_values = explainer(X_transformed)

    top_hospitals = (
        df_filtered
        .sort_values(by='boosted_score', ascending=False)
        .drop_duplicates(subset='hospital_name')
        .head(top_n)
    ).reset_index(drop=True)

    return top_hospitals, X_raw, shap_values, explainer

# ---------- Streamlit UI ----------
st.set_page_config(page_title="PM-JAY Hospital Recommender", layout="centered")
st.title("🏥 PM-JAY Hospital Recommender")
st.markdown("Find the best hospitals near you based on your **pincode** and disease.")

# Load geocoder with OpenCage
@st.cache_resource
def get_geocoder():
    return OpenCageGeocode("69f74547167a4d6e958d469ad0d940f6")  # Your API key

geocoder = get_geocoder()

# Disease selection
disease_list = sorted(df['disease_query'].dropna().unique())
selected_disease = st.selectbox("Select your disease/condition:", disease_list)

# Pincode input
user_location = st.text_input("📍 Enter your pincode:")

if user_location:
    try:
        results = geocoder.geocode(user_location + ", India")
        if results and len(results):
            geometry = results[0]['geometry']
            components = results[0]['components']
            user_lat = geometry['lat']
            user_lon = geometry['lng']
            user_state = components.get('state', '')
            full_address = results[0].get('formatted', '')

            st.success(f"📍 Location found: {full_address} ({user_lat:.4f}, {user_lon:.4f}) in **{user_state}**")

            if st.button("🔍 Recommend Hospitals") and user_state:
                results, X_raw, shap_values, explainer = recommend_hospitals(
                    user_lat, user_lon, user_state, selected_disease
                )

                if results is not None and not results.empty:
                    st.success(f"Top hospitals for **{selected_disease}** near you (preference for {user_state}):")
                    st.dataframe(results[['hospital_name', 'state', 'district', 'rating', 'distance_km', 'predicted_score']], use_container_width=True)

                    st.markdown("### 🔍 Feature Contributions (SHAP)")
                    for i in results.index:
                        st.markdown(f"**{results.at[i, 'hospital_name']}**")
                        fig, ax = plt.subplots()
                        shap.plots.waterfall(shap_values[i], show=False)
                        st.pyplot(fig)
                else:
                    st.error("No hospitals found for this disease.")
        else:
            st.warning("⚠️ Could not find the location. Try a valid pincode.")
    except Exception as e:
        st.error(f"❌ Geocoding failed: {e}")
else:
    st.info("Please enter your pincode to continue.")
