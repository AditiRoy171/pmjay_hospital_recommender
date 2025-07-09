import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians
from sklearn.metrics.pairwise import haversine_distances
from geopy.geocoders import Nominatim
import shap
import matplotlib.pyplot as plt

# ---------- Load Data and Model ----------
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_pmjay_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("hospital_xgb_pipeline.pkl")

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
st.markdown("Find the best hospitals near you based on your location and disease.")

# Disease selection
disease_list = sorted(df['disease_query'].dropna().unique())
selected_disease = st.selectbox("Select your disease/condition:", disease_list)

# Location input
user_location = st.text_input("📍 Enter your location (city or pincode):")

if user_location:
    try:
        geolocator = Nominatim(user_agent="hospital-recommender")
        location = geolocator.geocode(user_location)

        if location:
            user_lat = location.latitude
            user_lon = location.longitude
            user_state = None

            # Reverse geocoding for state
            reverse_location = geolocator.reverse((user_lat, user_lon), exactly_one=True)
            if reverse_location and 'state' in reverse_location.raw['address']:
                user_state = reverse_location.raw['address']['state']
                st.success(f"📍 Location found: {location.address} ({user_lat:.4f}, {user_lon:.4f}) in **{user_state}**")
            else:
                st.warning("⚠️ Could not detect your state for preference boosting.")

            # Recommend button
            if st.button("🔍 Recommend Hospitals") and user_state:
                results, X_raw, shap_values, explainer = recommend_hospitals(user_lat, user_lon, user_state, selected_disease)

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
            st.warning("⚠️ Could not find the location. Try being more specific.")
    except Exception as e:
        st.error(f"❌ Geocoding failed: {e}")
else:
    st.info("Please enter your location to continue.")
