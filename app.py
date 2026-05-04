import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="✈️ Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

# Load the trained model and feature names
@st.cache_resource
def load_model():
    model = joblib.load('flight_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, feature_names

model, feature_names = load_model()

# App title and description
st.title("✈️ Flight Delay Prediction System")
st.markdown("""
This app predicts whether a flight will be **delayed** or **on time** based on flight characteristics.
Built with a Random Forest classifier trained on real flight data. **Model Accuracy: 91.88%**
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📊 Model Info", "ℹ️ About"])

with tab1:
    st.header("Flight Information Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        departure_city = st.selectbox(
            "Departure City",
            ['Karachi', 'Islamabad', 'Lahore', 'Peshawar', 'Quetta', 'Multan'],
            help="Select the departure city"
        )
        
        route_type = st.selectbox(
            "Route Type",
            ['Domestic', 'International'],
            help="Domestic or International flight"
        )
        
        aircraft_type = st.selectbox(
            "Aircraft Type",
            ['Boeing 777', 'Boeing 787', 'Airbus A320', 'Airbus A350'],
            help="Type of aircraft operating the flight"
        )
    
    with col2:
        arrival_city = st.selectbox(
            "Arrival City",
            ['Dubai', 'London', 'Bangkok', 'Istanbul', 'Doha', 'Singapore'],
            help="Select the arrival city"
        )
        
        weather_condition = st.selectbox(
            "Weather Condition",
            ['Clear', 'Rainy', 'Cloudy', 'Stormy', 'Foggy'],
            help="Current weather condition at departure"
        )
        
        time_of_day = st.selectbox(
            "Time of Day",
            ['Morning (06:00-12:00)', 'Afternoon (12:00-18:00)', 'Evening (18:00-24:00)', 'Night (00:00-06:00)'],
            help="Scheduled departure time"
        )
    
    with col3:
        passenger_count = st.number_input(
            "Estimated Passengers",
            min_value=10,
            max_value=400,
            value=150,
            step=10,
            help="Expected number of passengers on board"
        )
        
        fuel_available = st.number_input(
            "Fuel Available (% capacity)",
            min_value=50,
            max_value=100,
            value=95,
            step=5,
            help="Fuel percentage available for the flight"
        )
        
        maintenance_days_ago = st.number_input(
            "Days Since Last Maintenance",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Days since last scheduled maintenance"
        )
    
    # Create feature input for prediction
    input_data = {
        'Departure_City': departure_city,
        'Arrival_City': arrival_city,
        'Route_Type': route_type,
        'Aircraft_Type': aircraft_type,
        'Weather_Condition': weather_condition,
        'Time_of_Day': time_of_day,
        'Passenger_Count': passenger_count,
        'Fuel_Available': fuel_available,
        'Maintenance_Days_Ago': maintenance_days_ago
    }
    
    # Create a sample dataframe for one-hot encoding
    sample_df = pd.DataFrame([input_data])
    
    # Predict button
    if st.button("🚀 Predict Flight Status", use_container_width=True):
        try:
            # One-hot encode the input
            sample_encoded = pd.get_dummies(sample_df, drop_first=True)
            
            # Align with training features
            for col in feature_names:
                if col not in sample_encoded.columns:
                    sample_encoded[col] = 0
            
            # Reorder columns to match training data
            sample_encoded = sample_encoded[feature_names]
            
            # Make prediction
            prediction = model.predict(sample_encoded)[0]
            probability = model.predict_proba(sample_encoded)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("📍 Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 'On Time':
                    st.success("✅ Flight is expected to be ON TIME", icon="✓")
                    confidence = probability[1] * 100
                else:
                    st.warning("⚠️ Flight is expected to be DELAYED", icon="!")
                    confidence = probability[0] * 100
                
                st.metric(
                    "Confidence Level",
                    f"{confidence:.1f}%",
                    delta="High" if confidence > 75 else "Medium" if confidence > 60 else "Low"
                )
            
            with col2:
                st.info(f"""
                **Flight Details Summary:**
                - Route: {departure_city} → {arrival_city}
                - Aircraft: {aircraft_type}
                - Passengers: {passenger_count}
                - Weather: {weather_condition}
                - Time: {time_of_day}
                """)
            
            # Additional info
            st.markdown("---")
            st.subheader("📊 Probability Distribution")
            prob_data = pd.DataFrame({
                'Status': ['On Time', 'Delayed'],
                'Probability': [probability[1] * 100, probability[0] * 100]
            })
            st.bar_chart(prob_data.set_index('Status'))
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check your inputs and try again.")

with tab2:
    st.header("🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        st.metric("Test Accuracy", "91.88%")
        st.metric("Algorithm", "Random Forest Classifier")
        st.metric("Total Features", "52 (after encoding)")
        st.metric("Training Samples", "640 (after SMOTE)")
    
    with col2:
        st.subheader("Dataset Statistics")
        st.info("""
        - **Total Flights**: 800
        - **Delayed Flights**: 746 (93.3%)
        - **On Time Flights**: 54 (6.7%)
        - **Balancing**: SMOTE applied to training data
        - **Train/Test Split**: 80/20
        """)
    
    st.subheader("Key Features Used")
    st.write("The model uses the following features for prediction:")
    features_display = pd.DataFrame({
        'Feature': feature_names[:10],
        'Type': ['Categorical'] * 10
    })
    st.dataframe(features_display, use_container_width=True)

with tab3:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## Flight Delay Prediction System
    
    This machine learning application predicts flight delays using a trained Random Forest classifier.
    
    ### How It Works
    1. **Input Flight Data**: Provide details about the flight including route, aircraft, weather, and other factors
    2. **Feature Processing**: The app converts categorical inputs into numerical format
    3. **Model Prediction**: The trained model analyzes the features and predicts the flight status
    4. **Results Display**: You get a prediction with confidence level
    
    ### Model Details
    - **Algorithm**: Random Forest Classifier with 100 trees
    - **Training Data**: 800 real flight records from PIA dataset
    - **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) used to handle imbalanced classes
    - **Accuracy**: 91.88% on test set
    
    ### Limitations
    - Predictions are based on historical patterns and may not account for all real-world factors
    - Weather predictions should be current for best results
    - Model is trained on specific route data; results may vary for routes not seen in training
    
    ### Built With
    - **Streamlit**: Interactive web framework
    - **scikit-learn**: Machine learning library
    - **pandas**: Data processing
    - **joblib**: Model serialization
    
    ### Version
    1.0 - Initial Release (May 2026)
    """)
