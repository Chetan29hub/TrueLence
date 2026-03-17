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

def show_home_page(predictor):
    """Display the home page with fake news detection functionality."""

    # Title and description
    st.markdown("""
    **Detect fake news using advanced machine learning algorithms.**

    Paste a news article or headline below and click "Check News" to analyze its authenticity.
    """)

    # Sidebar with information
    with st.sidebar:
        st.header("📚 History")
        st.markdown("**Recent Analyses:**")

        # Initialize history in session state if not exists
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []

        if st.session_state.analysis_history:
            # Show last 5 analyses
            for i, analysis in enumerate(st.session_state.analysis_history[-5:]):
                with st.expander(f"Analysis {len(st.session_state.analysis_history) - 4 + i}"):
                    st.write(f"**Text:** {analysis['text'][:100]}{'...' if len(analysis['text']) > 100 else ''}")
                    st.write(f"**Prediction:** {analysis['prediction']}")
                    st.write(f"**Confidence:** {int(analysis['confidence'] * 100)}%")
                    st.write(f"**Model:** {analysis['model_used']}")
                    st.write(f"**Time:** {analysis['timestamp']}")

            if st.button("🗑️ Clear History", key="clear_history"):
                st.session_state.analysis_history = []
                st.success("History cleared!")
                st.rerun()
        else:
            st.info("No analyses yet. Try analyzing some news content!")

        # Account section at the bottom
        st.markdown("---")
        st.header("👤 Account")

        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = {
                'name': 'Guest User',
                'email': 'guest@example.com'
            }

        st.write(f"**Name:** {st.session_state.user_profile['name']}")
        st.write(f"**Email:** {st.session_state.user_profile['email']}")

        if st.button("⚙️ Settings", key="account_settings"):
            st.info("Account settings would open here in a full application.")

        if st.button("🚪 Logout", key="logout"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Logged out successfully!")
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Enter News Content")

        # Text input area
        news_text = st.text_area(
            "Paste your news article or headline here:",
            height=200,
            placeholder="Enter the news content you want to analyze...",
            help="Enter any news article, headline, or text content for analysis."
        )

        # Model selection
        available_models = list(predictor.models.keys())
        selected_model = st.selectbox(
            "Select Prediction Model:",
            available_models,
            index=0 if available_models else None,
            help="Choose which machine learning model to use for prediction."
        )

        # Prediction button
        if st.button("🔍 Check News", type="primary", use_container_width=True):
            if not news_text.strip():
                st.error("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing news content..."):
                    # Add a small delay for better UX
                    time.sleep(1)

                    try:
                        # Make prediction
                        result = predictor.predict_single(news_text, selected_model)

                        # Store result in session state for display
                        st.session_state.last_result = result
                        st.session_state.show_results = True

                        # Add to history
                        if 'analysis_history' not in st.session_state:
                            st.session_state.analysis_history = []

                        history_entry = {
                            'text': news_text,
                            'prediction': result['prediction'],
                            'confidence': result['confidence'],
                            'model_used': result['model_used'],
                            'timestamp': time.strftime('%H:%M:%S')
                        }
                        st.session_state.analysis_history.append(history_entry)

                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

    with col2:
        st.subheader("🎯 Prediction Results")

        if 'show_results' in st.session_state and st.session_state.show_results:
            result = st.session_state.last_result

            # Prediction result box
            prediction_class = "real-news" if result['prediction'] == "Real News" else "fake-news"
            st.markdown(f"""
            <div class="prediction-box {prediction_class}">
                <h3 style="margin: 0; text-align: center;">{result['prediction']}</h3>
            </div>
            """, unsafe_allow_html=True)

            # Confidence score
            st.markdown("**Confidence Score:**")
            confidence_percentage = int(result['confidence'] * 100)

            # Confidence bar
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill {prediction_class.split('-')[0]}" style="width: {confidence_percentage}%"></div>
            </div>
            <p style="text-align: center; margin: 5px 0;">{confidence_percentage}%</p>
            """, unsafe_allow_html=True)

            # Detailed probabilities
            st.markdown("**Detailed Probabilities:**")
            prob_col1, prob_col2 = st.columns(2)

            with prob_col1:
                st.metric("Real News", f"{result['probability_real']:.1%}")

            with prob_col2:
                st.metric("Fake News", f"{result['probability_fake']:.1%}")

            # Model used
            st.markdown(f"**Model Used:** {result['model_used']}")

        else:
            # Placeholder when no results
            st.info("Enter news content and click 'Check News' to see results here.")

    # Additional features section
    st.markdown("---")
    st.subheader("🔄 Try Multiple Models")

    if st.button("Compare All Models", use_container_width=True):
        if not news_text.strip():
            st.error("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing with all models..."):
                time.sleep(1.5)

                try:
                    # Get predictions from all models
                    all_results = predictor.predict_multiple_models(news_text)

                    if all_results:
                        st.subheader("📊 Model Comparison")

                        # Create columns for each model
                        cols = st.columns(len(all_results))

                        for i, (model_name, result) in enumerate(all_results.items()):
                            with cols[i]:
                                # Model result card
                                prediction_class = "real-news" if result['prediction'] == "Real News" else "fake-news"

                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin: 0 0 10px 0; color: {'#22c55e' if result['prediction'] == 'Real News' else '#ef4444'};">{model_name}</h4>
                                    <div style="font-size: 18px; font-weight: bold; margin: 5px 0;">{result['prediction']}</div>
                                    <div style="font-size: 14px; color: #888;">Confidence: {int(result['confidence'] * 100)}%</div>
                                </div>
                                """, unsafe_allow_html=True)

                        # Add to history (store the first model's result as representative)
                        if 'analysis_history' not in st.session_state:
                            st.session_state.analysis_history = []

                        first_model_result = list(all_results.values())[0]
                        history_entry = {
                            'text': news_text,
                            'prediction': f"Multi-model: {first_model_result['prediction']}",
                            'confidence': first_model_result['confidence'],
                            'model_used': 'All Models',
                            'timestamp': time.strftime('%H:%M:%S')
                        }
                        st.session_state.analysis_history.append(history_entry)

                    else:
                        st.error("Could not get predictions from any model.")

                except Exception as e:
                    st.error(f"Error during multi-model prediction: {e}")

def main():
    """Main application function."""

    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Show login status
    if st.session_state.logged_in:
        st.success("✅ You are logged in")
    else:
        st.info("ℹ️ You are not logged in")

    # Load predictor
    predictor = load_predictor()

    if predictor is None:
        st.error("Failed to load prediction models. Please ensure models are trained and saved.")
        return

    # Main content based on current page
    if st.session_state.page == 'home':
        show_home_page(predictor)

if __name__ == "__main__":
    main()