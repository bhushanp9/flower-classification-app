import streamlit as st
import pickle
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

st.title("🌸 IRIS Flower Class Prediction")

# Load the model using st.cache_resource
@st.cache_resource
def load_model():
    with open("catboost.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Display model type
st.write(f"🔍 Model Type: `{type(model).__name__}`")

# Define label mapping (adjust if your labels are different)
label_map = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

# Get feature names from model or fallback to generic names
try:
    feature_names = model.feature_names_
    if not feature_names or all(f is None for f in feature_names):
        raise AttributeError
except AttributeError:
    feature_names = [f"feature_{i}" for i in range(model.feature_count_)]

# Input form
st.header("🧮 Input Features")
input_data = {}
for feature in feature_names:
    input_data[feature] = st.text_input(f"{feature}", "0")

# Predict button
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df.astype(float)

        prediction = model.predict(input_df)
        predicted_class = int(prediction[0])
        flower_name = label_map.get(predicted_class, "Unknown")

        # Show prediction
        st.success(f"✅ Predicted Flower: **{flower_name}**")

        # Show class probabilities if it's a classifier
        if isinstance(model, CatBoostClassifier):
            raw_prediction = model.predict_proba(input_df)
            st.write("📊 Class Probabilities:", raw_prediction)

    except ValueError as e:
        st.error(f"Invalid input: {e}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
