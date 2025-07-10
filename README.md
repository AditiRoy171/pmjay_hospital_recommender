# 🏥 PM-JAY Hospital Recommender

An AI-powered hospital recommendation system designed for **PM-JAY (Ayushman Bharat)** beneficiaries in India.

This tool helps users find the best hospitals near them based on their **pincode** and **disease/condition**. It ranks hospitals using an ML model trained on quality, proximity, and disease relevance — and explains every recommendation with **SHAP** (explainable AI).

👉 **[Try the App Live](https://pmjayhospitalrecommender-fujfcxp5rpmfmwvx8yte5j.streamlit.app/)**

---

## 🎯 Project Objective

Millions of patients under the PM-JAY scheme face challenges in accessing quality hospitals near them — especially in rural or remote regions.

This tool simplifies the process:
- Uses only a **pincode** for location input
- Filters hospitals based on the **user’s condition**
- Ranks options based on **quality**, **distance**, and **scheme-specific match**
- Applies **state preference boosting** to encourage local care
- Includes **transparent explainability** via SHAP waterfall plots

> ⚕️ A step toward **data-driven, equitable healthcare access** in India.

---

## 🚀 Features

✅ Pincode-based location detection  
✅ Disease-wise hospital filtering  
✅ Ranking based on model-predicted scores  
✅ State preference boosting  
✅ SHAP-based explainability  
✅ Fully deployed, no installation needed  

---

## 🧠 How It Works

The ML model predicts a **hospital suitability score** using:

| Feature         | Description                                 |
|----------------|---------------------------------------------|
| `rating`        | Hospital quality score                      |
| `distance_km`   | Distance from user to hospital              |
| `match_score`   | Whether the hospital treats the disease     |

### 🧮 Final Scoring Formula:

```text
base_score = model.predict([rating, distance_km, match_score])
boost_factor = 1.1 if hospital.state == user_state else 1.0
final_score = base_score × boost_factor

🔍 SHAP Explainability
Each hospital recommendation is accompanied by a SHAP waterfall plot showing:

How the model used each feature
Which attributes pushed the score up or down
Why that hospital is ranked highly
This ensures transparency and trust in the AI.

💻 Tech Stack

| Layer          | Tools Used                      |
| -------------- | ------------------------------- |
| Frontend UI    | Streamlit                       |
| ML Model       | XGBoost + Scikit-learn Pipeline |
| Geolocation    | OpenCage Geocoding API          |
| Explainability | SHAP                            |
| Deployment     | Streamlit Cloud                 |
| Data           | Synthetic PM-JAY dataset (CSV)  |

🌐 Live App
👉 Click to Try It Live
Enter a 6-digit pincode like 110001 (Delhi) or 832303 (Jharkhand), select a disease, and get your results!

- The synthetic dataset was created by sampling ~50 hospitals from the official PM-JAY hospital list and enriching them with logical features such as `rating`, `match_score`, and geolocations.
- Attributes were assigned based on realistic assumptions (e.g., higher ratings for urban hospitals, better disease match for specialty centers).
- This allowed training and testing the recommender without using any sensitive or restricted data.

## Final Remarks

This project demonstrates how data science and machine learning can be applied meaningfully to address real-world challenges in public healthcare access.

By leveraging synthetic data, explainable AI, and geospatial filtering, the system aims to support more transparent and informed decision-making for PM-JAY beneficiaries.

Feedback, suggestions, and collaborations are welcome. Thank you for taking the time to explore this work.

📄 License
This project is released under the MIT License — feel free to fork, modify, or build upon it with credit.
