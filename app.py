# app.py - Streamlit App for Automated Hotel Booking Cancellation Prediction

# Install required libraries in your environment:
# pip install streamlit pycaret pandas

import streamlit as st
import pandas as pd
from pycaret.classification import *

# Streamlit App Title
st.title("Automated Data Science: Hotel Booking Cancellation Predictor")

# Define the features used for training and prediction
used_features = [
    'hotel', 'lead_time', 'adults', 'children', 'babies', 'meal', 'market_segment',
    'distribution_channel', 'is_repeated_guest', 'deposit_type', 'adr',
    'total_of_special_requests', 'required_car_parking_spaces'
]

# Step 1: Upload or use default dataset
st.header("Step 1: Upload Cleaned Dataset")
st.info("Upload your cleaned CSV from PandaBI or use our default 5000-row dataset.")
uploaded_file = st.file_uploader("Upload your cleaned CSV file (e.g., cleaned_hotel_bookings.csv)", type="csv")

# Load dataset (default if no file uploaded)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    source = "Uploaded CSV"
else:
    default_url = "https://github.com/MisterArbazzz/AutoDS/blob/master/cleaned_hotel_data.csv"  # Replace with your actual URL
    df = pd.read_csv(default_url)
    source = "Default 5000-row dataset"
    st.warning("No file uploaded. Using default pre-cleaned dataset with 5000 rows.")

st.write(f"Dataset Source: {source}")
st.write("Dataset Preview:")
st.dataframe(df.head())
st.write(f"Dataset Shape: {df.shape}")

# Option to sample for speed
sample_size = st.slider("Sample Size for Quick AutoML (to speed up)", min_value=1000, max_value=len(df), value=min(5000, len(df)), step=1000)
df_sampled = df.sample(n=sample_size, random_state=42) if sample_size < len(df) else df
st.write(f"Using sampled dataset of {df_sampled.shape[0]} rows for AutoML.")

# Step 2: Run AutoML with PyCaret
st.header("Step 2: Run AutoML")
if st.button("Run AutoML"):
    with st.spinner("Running AutoML... This may take a few minutes."):
        # Compute features to ignore (all columns not in used_features or target)
        all_columns = df_sampled.columns.tolist()
        ignore_features = [col for col in all_columns if col not in used_features + ['is_canceled']]

        # Setup PyCaret with ignored features to avoid KeyError
        clf = setup(data=df_sampled, target='is_canceled', train_size=0.8,
                    normalize=True, remove_multicollinearity=True,
                    ignore_features=ignore_features,
                    session_id=42, verbose=False, html=False)

        # Compare limited models for speed
        best_models = compare_models(include=['lightgbm', 'rf'], fold=5, sort='AUC', turbo=True)

        # Get the best model
        best_model_name = pull().index[0]
        best_model = create_model(best_model_name)
        tuned_best_model = tune_model(best_model, fold=5)

        # Display results
        st.success("AutoML Completed!")
        st.subheader("Model Comparison Results")
        st.dataframe(pull())

        st.subheader("Best Model Evaluation")
        st.write(f"Best Model: {best_model_name}")
        auc_plot = plot_model(tuned_best_model, plot='auc', save=True)
        st.image('AUC.png')
        feature_plot = plot_model(tuned_best_model, plot='feature', save=True)
        st.image('Feature Importance.png')

        # Save the best model in session state
        final_model = finalize_model(tuned_best_model)
        st.session_state['model'] = final_model
        st.write("Best model selected and ready for predictions!")

# Step 3: Make Predictions with the Best Model
if 'model' in st.session_state:
    st.header("Step 3: Make Predictions")
    model = st.session_state['model']

    # Input fields for key features
    hotel = st.selectbox("Hotel Type", ['Resort Hotel', 'City Hotel'])
    lead_time = st.number_input("Lead Time (days)", min_value=0, value=0)
    adults = st.number_input("Adults", min_value=0, value=1)
    children = st.number_input("Children", min_value=0, value=0)
    babies = st.number_input("Babies", min_value=0, value=0)
    meal = st.selectbox("Meal", ['BB', 'FB', 'HB', 'SC', 'Undefined'])
    market_segment = st.selectbox("Market Segment", ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Groups', 'Complementary', 'Aviation'])
    distribution_channel = st.selectbox("Distribution Channel", ['Direct', 'Corporate', 'TA/TO', 'GDS'])
    is_repeated_guest = st.checkbox("Is Repeated Guest?")
    deposit_type = st.selectbox("Deposit Type", ['No Deposit', 'Non Refund', 'Refundable'])
    adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, value=0.0)
    total_of_special_requests = st.number_input("Total Special Requests", min_value=0, value=0)
    required_car_parking_spaces = st.number_input("Required Parking Spaces", min_value=0, value=0)

    if st.button("Predict Cancellation"):
        # Create input DataFrame
        input_data = pd.DataFrame({
            'hotel': [hotel],
            'lead_time': [lead_time],
            'adults': [adults],
            'children': [children],
            'babies': [babies],
            'meal': [meal],
            'market_segment': [market_segment],
            'distribution_channel': [distribution_channel],
            'is_repeated_guest': [1 if is_repeated_guest else 0],
            'deposit_type': [deposit_type],
            'adr': [adr],
            'total_of_special_requests': [total_of_special_requests],
            'required_car_parking_spaces': [required_car_parking_spaces]
        })

        # Predict
        prediction = predict_model(model, data=input_data)
        label = prediction['prediction_label'].iloc[0]
        score = prediction['prediction_score'].iloc[0] if 'prediction_score' in prediction else prediction['Score'].iloc[0]

        st.success(f"Prediction: {'Canceled' if label == 1 else 'Not Canceled'} (Confidence: {score:.2f})")
else:
    st.info("Upload dataset or use default and run AutoML to enable predictions.")