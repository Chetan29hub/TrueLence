#!/usr/bin/env python3
"""
Fake News Detection - Streamlit Web Application

A simple web application for detecting fake news using machine learning models.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.prediction import FakeNewsPredictor
import time

# Configure page settings
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load the fake news predictor model."""
    try:
        predictor = FakeNewsPredictor()
        return predictor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

predictor = load_predictor()

# Title
st.title("📰 Fake News Detection")
st.markdown("Analyze news content to determine if it's real or fake using AI.")

# Sidebar
st.sidebar.header("📊 Model Information")
if predictor:
    model_info = predictor.get_model_info()
    st.sidebar.write(f"**Available Models:** {len(model_info['available_models'])}")
    st.sidebar.write("**Models:**")
    for model in model_info['available_models']:
        st.sidebar.write(f"- {model}")

# Main content
st.header("🔍 News Analysis")

# Text input
news_text = st.text_area(
    "Enter news content to analyze:",
    height=150,
    placeholder="Paste your news article, headline, or any text content here..."
)

# Model selection
model_options = ["Single Model (Logistic Regression)", "Compare All Models"]
selected_model = st.selectbox("Select analysis type:", model_options)

# Analyze button
if st.button("🔍 Analyze News", type="primary"):
    if not news_text.strip():
        st.error("Please enter some text to analyze.")
    elif not predictor:
        st.error("Model loading failed. Please check the logs.")
    else:
        with st.spinner("Analyzing content..."):
            time.sleep(1)  # Simulate processing time

            try:
                if selected_model == "Single Model (Logistic Regression)":
                    # Single model prediction
                    result = predictor.predict_single(news_text)

                    # Display result
                    st.success("Analysis Complete!")

                    col1, col2 = st.columns(2)

                    with col1:
                        if result['prediction'] == 'Real News':
                            st.markdown("### ✅ REAL NEWS")
                            st.markdown(f"**Confidence:** {result['confidence']:.1%}")
                        else:
                            st.markdown("### ❌ FAKE NEWS")
                            st.markdown(f"**Confidence:** {result['confidence']:.1%}")

                    with col2:
                        # Confidence meter
                        confidence_pct = result['confidence'] * 100
                        st.markdown("**Confidence Level:**")
                        st.progress(result['confidence'])

                        if confidence_pct > 80:
                            st.markdown("🎯 **High Confidence**")
                        elif confidence_pct > 60:
                            st.markdown("⚠️ **Medium Confidence**")
                        else:
                            st.markdown("🤔 **Low Confidence**")

                    # Detailed results
                    with st.expander("📊 Detailed Results"):
                        st.write(f"**Model Used:** {result['model_used']}")
                        st.write(f"**Prediction:** {result['prediction']}")
                        st.write(".1%")
                        st.write(".1%")

                else:
                    # Multiple models comparison
                    results = predictor.predict_multiple_models(news_text)

                    st.success("Multi-Model Analysis Complete!")

                    # Display results for each model
                    st.subheader("🤖 Model Comparison")

                    for model_name, result in results.items():
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])

                            with col1:
                                st.markdown(f"**{model_name}**")

                            with col2:
                                if result['prediction'] == 'Real News':
                                    st.markdown("✅ Real")
                                else:
                                    st.markdown("❌ Fake")

                            with col3:
                                st.markdown(".1%")

                            # Progress bar
                            st.progress(result['confidence'])

                            st.markdown("---")

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and scikit-learn | 🧠 AI-Powered Fake News Detection")