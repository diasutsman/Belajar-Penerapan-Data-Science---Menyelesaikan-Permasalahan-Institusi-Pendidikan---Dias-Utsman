import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Set page configuration
st.set_page_config(
    page_title="Jaya Jaya Institut - Student Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
    }
    .section-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1rem;
    }
    .insight-text {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #3B82F6;
    }
    .prediction-box {
        background-color: #F0FDF4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #10B981;
        margin-top: 1rem;
    }
    .warning-box {
        background-color: #FEF2F2;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #EF4444;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/dropout_prediction_model.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Try alternative file extension as fallback
        # try:
        #     model = joblib.load('model/dropout_prediction_model.pkl')
        #     return model
        # except Exception as e2:
        #     st.error(f"Error loading fallback model: {e2}")
        return None

@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load('model/scaler.joblib')
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        # Try alternative file extension as fallback
        # try:
        #     scaler = joblib.load('model/scaler.pkl')
        #     return scaler
        # except Exception as e2:
        #     st.error(f"Error loading fallback scaler: {e2}")
        return None

@st.cache_data
def load_data():
    df = pd.read_csv('data.csv', sep=';')
    return df

# Load data and model
try:
    model = load_model()
    scaler = load_scaler()
    if model is None or scaler is None:
        st.warning("⚠️ Some model components failed to load. Prediction functionality may be limited.")
    df = load_data()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model or data: {e}")

# Sidebar
st.sidebar.markdown("<h1 class='main-header'>Jaya Jaya Institut</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 class='sub-header'>Student Dropout Prediction</h2>", unsafe_allow_html=True)
st.sidebar.image("https://img.freepik.com/free-vector/flat-design-university-concept_23-2148192778.jpg", use_container_width=True)

pages = ["Dashboard", "Student Predictor", "About"]
selection = st.sidebar.radio("Navigation", pages)

# Dashboard page
if selection == "Dashboard":
    st.markdown("<h1 class='main-header'>Jaya Jaya Institut - Student Performance Dashboard</h1>", unsafe_allow_html=True)
    
    # Display dataset overview
    st.markdown("<h2 class='section-header'>Student Dataset Overview</h2>", unsafe_allow_html=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_students = len(df)
        st.metric("Total Students", f"{total_students:,}")
    
    with col2:
        dropout_rate = df[df['Status'] == 'Dropout'].shape[0] / total_students * 100
        st.metric("Dropout Rate", f"{dropout_rate:.2f}%")
    
    with col3:
        graduation_rate = df[df['Status'] == 'Graduate'].shape[0] / total_students * 100
        st.metric("Graduation Rate", f"{graduation_rate:.2f}%")
    
    # Status distribution chart
    st.markdown("<h3 class='section-header'>Student Status Distribution</h3>", unsafe_allow_html=True)
    status_counts = df['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    fig = px.pie(status_counts, values='Count', names='Status', 
                 color_discrete_sequence=px.colors.qualitative.Set3,
                 title='Distribution of Student Status')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Student performance metrics
    st.markdown("<h3 class='section-header'>Student Performance Metrics</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        # First semester performance by status
        sem1_perf = df.groupby('Status')['Curricular_units_1st_sem_approved'].mean().reset_index()
        fig = px.bar(sem1_perf, x='Status', y='Curricular_units_1st_sem_approved',
                     color='Status', color_discrete_sequence=px.colors.qualitative.Pastel,
                     title='Average Units Approved in 1st Semester by Status')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Second semester performance by status
        sem2_perf = df.groupby('Status')['Curricular_units_2nd_sem_approved'].mean().reset_index()
        fig = px.bar(sem2_perf, x='Status', y='Curricular_units_2nd_sem_approved',
                     color='Status', color_discrete_sequence=px.colors.qualitative.Pastel,
                     title='Average Units Approved in 2nd Semester by Status')
        st.plotly_chart(fig, use_container_width=True)
    
    # Key factors affecting dropout
    st.markdown("<h3 class='section-header'>Key Factors Affecting Student Dropout</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Admission grade by status
        fig = px.box(df, x='Status', y='Admission_grade', color='Status',
                     title='Admission Grade by Student Status',
                     color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age distribution by status
        fig = px.histogram(df, x='Age_at_enrollment', color='Status',
                           title='Age Distribution by Student Status',
                           color_discrete_sequence=px.colors.qualitative.Safe,
                           barmode='overlay', opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation between key features
    st.markdown("<h3 class='section-header'>Correlation Between Key Features</h3>", unsafe_allow_html=True)
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create a heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, center=0, ax=ax)
    st.pyplot(fig)
    
    # Insights
    st.markdown("<h3 class='section-header'>Key Insights</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-text'>
    <ul>
        <li>There is a significant correlation between first semester performance and dropout rates</li>
        <li>Students with lower admission grades have higher tendency to dropout</li>
        <li>Age at enrollment shows some correlation with student success</li>
        <li>Economic factors such as being a scholarship holder or having debts impact student persistence</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Student Predictor page
elif selection == "Student Predictor":
    st.markdown("<h1 class='main-header'>Student Dropout Prediction Tool</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-text'>
    This tool allows you to predict whether a student is at risk of dropping out based on various factors.
    Fill in the student information below to get a prediction.
    </div>
    """, unsafe_allow_html=True)
    
    if not model_loaded:
        st.warning("Model is not loaded. Please check the model files.")
    else:
        # Create three columns for input fields
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<h3 class='section-header'>Demographic Information</h3>", unsafe_allow_html=True)
            
            gender = st.selectbox("Gender", options=["Male", "Female"], 
                                 format_func=lambda x: x)
            
            age = st.slider("Age at Enrollment", min_value=16, max_value=70, value=20)
            
            marital_status = st.selectbox("Marital Status", 
                                        options=["Single", "Married", "Widower", "Divorced", "Facto union", "Legally separated"],
                                        index=0)
            
            nationality = st.selectbox("Nationality", 
                                     options=["Portuguese", "Other"],
                                     index=0)
            
            displaced = st.selectbox("Displaced from Home", 
                                   options=["Yes", "No"],
                                   index=1)
            
            special_needs = st.selectbox("Educational Special Needs", 
                                       options=["Yes", "No"],
                                       index=1)
        
        with col2:
            st.markdown("<h3 class='section-header'>Academic Background</h3>", unsafe_allow_html=True)
            
            application_mode = st.selectbox("Application Mode", 
                                          options=["General contingent", "Special contingent", "Over 23 years old", "Transfer", "Other"],
                                          index=0)
            
            course = st.selectbox("Course Type", 
                                options=["Sciences", "Technologies", "Arts", "Management", "Social Service", "Health"],
                                index=0)
            
            daytime_evening = st.selectbox("Daytime/Evening Attendance", 
                                         options=["Daytime", "Evening"],
                                         index=0)
            
            prev_qualification = st.selectbox("Previous Qualification", 
                                            options=["Secondary education", "Higher education", "Frequency of higher education", "Other"],
                                            index=0)
            
            prev_qualification_grade = st.slider("Previous Qualification Grade", min_value=0, max_value=200, value=130)
            
            admission_grade = st.slider("Admission Grade", min_value=0, max_value=200, value=130)
        
        with col3:
            st.markdown("<h3 class='section-header'>Economic & Performance Factors</h3>", unsafe_allow_html=True)
            
            scholarship = st.selectbox("Scholarship Holder", 
                                     options=["Yes", "No"],
                                     index=1)
            
            debtor = st.selectbox("Debtor", 
                                options=["Yes", "No"],
                                index=1)
            
            tuition_fees_up_to_date = st.selectbox("Tuition Fees Up To Date", 
                                                 options=["Yes", "No"],
                                                 index=0)
            
            units_1st_sem_enrolled = st.number_input("Curricular Units 1st Sem (Enrolled)", min_value=0, max_value=10, value=6)
            
            units_1st_sem_approved = st.number_input("Curricular Units 1st Sem (Approved)", min_value=0, max_value=10, value=5)
            
            units_1st_sem_grade = st.slider("Curricular Units 1st Sem (Grade)", min_value=0.0, max_value=20.0, value=12.0)

        # Function to prepare input for prediction
        def prepare_input_data():
            # Transform categorical inputs to match the model's expected encoding
            gender_encoded = 1 if gender == "Male" else 0
            displaced_encoded = 1 if displaced == "Yes" else 0
            special_needs_encoded = 1 if special_needs == "Yes" else 0
            scholarship_encoded = 1 if scholarship == "Yes" else 0
            debtor_encoded = 1 if debtor == "Yes" else 0
            tuition_encoded = 1 if tuition_fees_up_to_date == "Yes" else 0
            daytime_encoded = 1 if daytime_evening == "Daytime" else 0
            nationality_encoded = 1 if nationality == "Portuguese" else 2
            
            # Map marital status to encoding
            marital_mapping = {"Single": 1, "Married": 2, "Widower": 3, "Divorced": 4, "Facto union": 5, "Legally separated": 6}
            marital_encoded = marital_mapping.get(marital_status, 1)
            
            # Map application mode to simplified encoding
            application_mapping = {"General contingent": 1, "Special contingent": 2, "Over 23 years old": 39, "Transfer": 42, "Other": 27}
            application_encoded = application_mapping.get(application_mode, 1)
            
            # Map course to simplified encoding
            course_mapping = {"Sciences": 9003, "Technologies": 9119, "Arts": 9070, "Management": 9147, "Social Service": 9238, "Health": 9500}
            course_encoded = course_mapping.get(course, 9003)
            
            # Map previous qualification to simplified encoding
            prev_qual_mapping = {"Secondary education": 1, "Higher education": 2, "Frequency of higher education": 6, "Other": 19}
            prev_qual_encoded = prev_qual_mapping.get(prev_qualification, 1)
            
            # Create input features
            input_data = {
                'Marital_status': marital_encoded,
                'Application_mode': application_encoded,
                'Course': course_encoded,
                'Daytime_evening_attendance': daytime_encoded,
                'Previous_qualification': prev_qual_encoded,
                'Previous_qualification_grade': prev_qualification_grade,
                'Nacionality': nationality_encoded,
                'Admission_grade': admission_grade,
                'Displaced': displaced_encoded,
                'Educational_special_needs': special_needs_encoded,
                'Debtor': debtor_encoded,
                'Tuition_fees_up_to_date': tuition_encoded,
                'Gender': gender_encoded,
                'Scholarship_holder': scholarship_encoded,
                'Age_at_enrollment': age,
                'Curricular_units_1st_sem_enrolled': units_1st_sem_enrolled,
                'Curricular_units_1st_sem_approved': units_1st_sem_approved,
                'Curricular_units_1st_sem_grade': units_1st_sem_grade
            }
            
            return input_data
        
        # Make prediction when the button is clicked
        if st.button("Predict Dropout Risk"):
            input_data = prepare_input_data()
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure we have all required columns
            expected_columns = scaler.feature_names_in_
            missing_cols = set(expected_columns) - set(input_df.columns)
            
            # Handle missing columns
            for col in missing_cols:
                input_df[col] = 0  # Fill with default value
            
            # Select and order columns correctly
            input_df = input_df[expected_columns]
            
            # Scale the features
            input_scaled = scaler.transform(input_df)
            
            # Get prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Determine highest probability class
            highest_prob_index = np.argmax(prediction_proba)
            highest_prob = prediction_proba[highest_prob_index]
            
            # Map to status labels
            status_labels = ['Dropout', 'Enrolled', 'Graduate']
            predicted_status = status_labels[highest_prob_index]
            
            # Display prediction
            if predicted_status == 'Dropout':
                st.markdown(f"""
                <div class='warning-box'>
                <h2>⚠️ High Dropout Risk: {highest_prob*100:.2f}%</h2>
                <p>This student shows significant risk factors for dropping out. Consider immediate intervention.</p>
                <h3>Recommended Actions:</h3>
                <ul>
                    <li>Schedule academic counseling session</li>
                    <li>Provide additional tutoring for difficult subjects</li>
                    <li>Evaluate financial support options if applicable</li>
                    <li>Connect with student mentorship program</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            elif predicted_status == 'Enrolled':
                st.markdown(f"""
                <div class='prediction-box'>
                <h2>🔍 Moderate Risk: {highest_prob*100:.2f}%</h2>
                <p>This student appears to be on track but may benefit from some additional support.</p>
                <h3>Suggested Actions:</h3>
                <ul>
                    <li>Regular check-ins with academic advisor</li>
                    <li>Encourage participation in study groups</li>
                    <li>Monitor performance in upcoming assessments</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:  # Graduate
                st.markdown(f"""
                <div class='prediction-box'>
                <h2>✅ Low Dropout Risk: {highest_prob*100:.2f}%</h2>
                <p>This student shows strong indicators of academic success and is likely to graduate.</p>
                <h3>Suggested Actions:</h3>
                <ul>
                    <li>Provide information about advanced courses and opportunities</li>
                    <li>Encourage participation in leadership and extracurricular activities</li>
                    <li>Discuss long-term career planning and post-graduation options</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Display probability distribution
            st.markdown("<h3 class='section-header'>Prediction Probability Distribution</h3>", unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({
                'Status': status_labels,
                'Probability': prediction_proba
            })
            
            fig = px.bar(prob_df, x='Status', y='Probability', color='Status',
                        color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(xaxis_title='Student Status', yaxis_title='Probability')
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (simplified explanation)
            st.markdown("<h3 class='section-header'>Key Factors for This Prediction</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div class='insight-text'>
            <p>The most important factors affecting dropout prediction are typically:</p>
            <ol>
                <li>First semester academic performance (courses approved and grades)</li>
                <li>Admission grade</li>
                <li>Age at enrollment</li>
                <li>Financial factors (scholarship status, debts)</li>
                <li>Previous qualification grade</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

# About page
else:
    st.markdown("<h1 class='main-header'>About This Project</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='insight-text'>
    <h3>Project Overview</h3>
    <p>This project was developed to help Jaya Jaya Institut identify students at risk of dropping out early in their academic journey. 
    By using machine learning algorithms, we can predict which students might need additional support and intervention.</p>
    
    <h3>Data Source</h3>
    <p>The dataset used in this project contains information about students, including demographics, academic performance, and socio-economic factors.</p>
    
    <h3>Model Information</h3>
    <p>We developed a machine learning model that analyzes various student factors to predict their likelihood of dropping out. 
    The model has been trained on historical data from Jaya Jaya Institut and validated for accuracy.</p>
    
    <h3>How to Use This Tool</h3>
    <p>Administrators and academic advisors can use this tool to:</p>
    <ul>
        <li>Explore patterns in student data through the Dashboard</li>
        <li>Input information about specific students to predict their dropout risk</li>
        <li>Identify key factors that contribute to student success or failure</li>
        <li>Develop targeted intervention strategies based on risk assessments</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='section-header'>Development Team</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-text'>
    <p>This application was developed as part of a data science project to help educational institutions reduce student dropout rates.</p>
    </div>
    """, unsafe_allow_html=True)