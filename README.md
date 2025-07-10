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


That last line — ```` ``` ```` — is what **closes the Python code block**.

---

### ✅ Full example with section separator:

```markdown
### 🧮 Final Scoring Formula

```python
base_score = model.predict([rating, distance_km, match_score])
boost_factor = 1.1 if hospital.state == user_state else 1.0
final_score = base_score × boost_factor

## 🔍 SHAP Explainability

This project uses **SHAP (SHapley Additive exPlanations)** to make model predictions transparent and interpretable.

For each recommended hospital, a **SHAP waterfall plot** is generated to show:

- 📈 How each feature (like `rating`, `distance_km`, `match_score`) contributed to the final score  
- 📉 Which features decreased the score and by how much  
- 🧠 Why a hospital ranked higher than others for a given query  

These visual explanations ensure that users — whether patients or stakeholders — can trust the **"why"** behind every recommendation.

> SHAP makes machine learning models explainable, fair, and more human-understandable — a critical step when building for public good.

---

## 💻 Tech Stack

| Layer          | Tools Used                      |
|----------------|---------------------------------|
| Frontend UI    | Streamlit                       |
| ML Model       | XGBoost + Scikit-learn Pipeline |
| Geolocation    | OpenCage Geocoding API          |
| Explainability | SHAP                            |
| Deployment     | Streamlit Cloud                 |
| Data           | Synthetic PM-JAY dataset (CSV)  |

---

## 📦 Installation (for local testing)

```bash
git clone https://github.com/your-username/pmjay_hospital_recommender.git
cd pmjay_hospital_recommender
pip install -r requirements.txt
streamlit run app.py
