"""
Wine Quality Prediction - Streamlit Web Application
Interactive web app for wine quality classification using multiple ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #8B0000;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4A4A4A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #8B0000;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🍷 Wine Quality Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning Model Comparison Dashboard</p>', unsafe_allow_html=True)

# Load models function
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'lr_model.pkl',
        'Decision Tree': 'dt_model.pkl',
        'K-Nearest Neighbors': 'knn_model.pkl',
        'Naive Bayes': 'nb_model.pkl',
        'Random Forest': 'rf_model.pkl',
        'XGBoost': 'xgb_model.pkl'
    }

    for name, file in model_files.items():
        try:
            with open(file, 'rb') as f:
                models[name] = pickle.load(f)
        except:
            st.warning(f"Could not load {name} model")

    # Load scaler
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = None

    return models, scaler

# Load pre-computed results
@st.cache_data
def load_results():
    try:
        results_df = pd.read_csv('model_results.csv')
        return results_df
    except:
        return None

# Sidebar
st.sidebar.header("📊 Dashboard Controls")
st.sidebar.markdown("---")

# Model selection
st.sidebar.subheader("🤖 Model Selection")
models, scaler = load_models()
model_names = list(models.keys())
selected_model = st.sidebar.selectbox(
    "Choose a model:",
    model_names,
    index=0
)

st.sidebar.markdown("---")

# Upload option
st.sidebar.subheader("📁 Upload Test Data")
st.sidebar.info("Upload a CSV file with wine features (without quality column)")
uploaded_file = st.sidebar.file_uploader(
    "Choose CSV file",
    type=['csv'],
    help="Upload test data in CSV format"
)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Comparison", "🎯 Model Performance", "🔮 Make Predictions", "📚 About Dataset"])

with tab1:
    st.header("Model Comparison - All Metrics")

    results_df = load_results()

    if results_df is not None:
        # Display comparison table
        st.subheader("📊 Performance Metrics Comparison")

        # Style the dataframe
        styled_df = results_df.style.highlight_max(
            subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
            color='lightgreen'
        ).format({
            'Accuracy': '{:.4f}',
            'AUC': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1': '{:.4f}',
            'MCC': '{:.4f}'
        })

        st.dataframe(styled_df, use_container_width=True)

        # Visualization
        st.subheader("📉 Visual Comparison")

        col1, col2 = st.columns(2)

        with col1:
            # Bar chart for accuracy
            fig_acc = px.bar(
                results_df,
                x='Model',
                y='Accuracy',
                title='Accuracy Comparison',
                color='Accuracy',
                color_continuous_scale='Reds',
                text='Accuracy'
            )
            fig_acc.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_acc.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_acc, use_container_width=True)

        with col2:
            # Bar chart for F1 Score
            fig_f1 = px.bar(
                results_df,
                x='Model',
                y='F1',
                title='F1 Score Comparison',
                color='F1',
                color_continuous_scale='Blues',
                text='F1'
            )
            fig_f1.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_f1.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_f1, use_container_width=True)

        # Radar chart for all metrics
        st.subheader("🕸️ Multi-Metric Radar Chart")

        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']

        fig_radar = go.Figure()

        for idx, row in results_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=metrics,
                fill='toself',
                name=row['Model']
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="All Models - Multi-Metric Comparison"
        )

        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("Pre-computed results not found. Please run model training first.")

with tab2:
    st.header(f"🎯 {selected_model} - Detailed Performance")

    if results_df is not None:
        model_result = results_df[results_df['Model'] == selected_model].iloc[0]

        # Display metrics in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", f"{model_result['Accuracy']:.4f}")
            st.metric("Precision", f"{model_result['Precision']:.4f}")

        with col2:
            st.metric("AUC Score", f"{model_result['AUC']:.4f}")
            st.metric("Recall", f"{model_result['Recall']:.4f}")

        with col3:
            st.metric("F1 Score", f"{model_result['F1']:.4f}")
            st.metric("MCC Score", f"{model_result['MCC']:.4f}")

        st.markdown("---")

        # Load test data for confusion matrix
        try:
            test_data = pd.read_csv('test_data.csv')
            X_test = test_data.drop('quality', axis=1)
            y_test = test_data['quality']

            # Get predictions
            model = models[selected_model]
            y_pred = model.predict(X_test)

            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Bad Quality', 'Good Quality'],
                y=['Bad Quality', 'Good Quality'],
                color_continuous_scale='Reds',
                text_auto=True,
                title=f"Confusion Matrix - {selected_model}"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # Classification Report
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, target_names=['Bad Quality', 'Good Quality'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

        except Exception as e:
            st.warning(f"Could not load test data: {e}")

with tab3:
    st.header("🔮 Make Predictions on New Data")

    if uploaded_file is not None:
        try:
            # Read uploaded file
            df_upload = pd.read_csv(uploaded_file)

            st.success(f"✅ File uploaded successfully! Shape: {df_upload.shape}")

            # Display first few rows
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(df_upload.head(), use_container_width=True)

            # Make predictions
            if st.button("🚀 Run Predictions", type="primary"):
                if scaler is not None and selected_model in models:
                    # Scale the data
                    X_scaled = scaler.transform(df_upload)

                    # Get model
                    model = models[selected_model]

                    # Predictions
                    predictions = model.predict(X_scaled)
                    predictions_proba = model.predict_proba(X_scaled)

                    # Add predictions to dataframe
                    results = df_upload.copy()
                    results['Predicted_Quality'] = ['Good Quality' if p == 1 else 'Bad Quality' for p in predictions]
                    results['Confidence'] = [max(proba) for proba in predictions_proba]

                    st.subheader("🎉 Prediction Results")
                    st.dataframe(results, use_container_width=True)

                    # Summary
                    st.subheader("📊 Prediction Summary")
                    col1, col2 = st.columns(2)

                    with col1:
                        good_count = sum(predictions == 1)
                        bad_count = sum(predictions == 0)
                        st.metric("Good Quality Wines", good_count)
                        st.metric("Bad Quality Wines", bad_count)

                    with col2:
                        # Pie chart
                        fig_pie = px.pie(
                            values=[good_count, bad_count],
                            names=['Good Quality', 'Bad Quality'],
                            title='Quality Distribution',
                            color_discrete_sequence=['#2ecc71', '#e74c3c']
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    # Download results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Model or scaler not loaded properly")

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("👆 Please upload a CSV file using the sidebar")

        # Show example format
        st.subheader("📋 Expected Data Format")
        st.markdown("""
        Your CSV should contain the following columns:
        - fixed acidity
        - volatile acidity
        - citric acid
        - residual sugar
        - chlorides
        - free sulfur dioxide
        - total sulfur dioxide
        - density
        - pH
        - sulphates
        - alcohol

        **Do not include the 'quality' column** - the model will predict it!
        """)

with tab4:
    st.header("📚 About the Wine Quality Dataset")

    st.markdown("""
    ### Dataset Information

    The **Wine Quality Dataset** contains physicochemical properties of red wine variants of Portuguese "Vinho Verde" wine.

    #### 📊 Dataset Statistics
    - **Total Samples**: 1,599 wines
    - **Features**: 11 physicochemical properties
    - **Target**: Quality score (0-10 scale, converted to binary: Good/Bad)

    #### 🔬 Features Description

    1. **Fixed Acidity**: Most acids involved with wine or fixed or nonvolatile (g/dm³)
    2. **Volatile Acidity**: Amount of acetic acid in wine (g/dm³)
    3. **Citric Acid**: Found in small quantities, adds freshness (g/dm³)
    4. **Residual Sugar**: Amount of sugar remaining after fermentation (g/dm³)
    5. **Chlorides**: Amount of salt in the wine (g/dm³)
    6. **Free Sulfur Dioxide**: Prevents microbial growth (mg/dm³)
    7. **Total Sulfur Dioxide**: Amount of free and bound forms (mg/dm³)
    8. **Density**: Density of wine (g/cm³)
    9. **pH**: Acidity or basicity level (0-14 scale)
    10. **Sulphates**: Wine additive (g/dm³)
    11. **Alcohol**: Alcohol percentage (% by volume)

    #### 🎯 Classification Task

    **Binary Classification**: Wines with quality rating ≥ 6 are classified as **Good Quality**, 
    while wines with rating < 6 are classified as **Bad Quality**.

    #### 📖 Source

    P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
    Modeling wine preferences by data mining from physicochemical properties. 
    In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

    **UCI Machine Learning Repository**
    """)

    # Dataset statistics visualization
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=';')

        st.subheader("📈 Feature Distributions")

        # Select feature to visualize
        feature = st.selectbox("Select feature to visualize:", df.columns[:-1])

        fig_hist = px.histogram(
            df,
            x=feature,
            title=f'Distribution of {feature}',
            color_discrete_sequence=['#8B0000']
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    except:
        st.info("Feature distributions will be shown here once data is loaded")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🍷 Wine Quality Classification Dashboard | Built with Streamlit & Scikit-learn</p>
        <p>M.Tech (AIML/DSE) - Machine Learning Assignment 2</p>
    </div>
    """, unsafe_allow_html=True)
