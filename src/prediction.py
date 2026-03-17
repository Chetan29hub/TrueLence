"""
Fake News Detection - Prediction Module

This module contains functions for making predictions on new text data
using trained machine learning models.
"""

import joblib
import numpy as np
from .preprocessing import TextPreprocessor

class FakeNewsPredictor:
    """
    A class for making predictions with trained fake news detection models.
    """

    def __init__(self, models_dir='models'):
        """
        Initialize the predictor by loading trained models.

        Args:
            models_dir (str): Directory containing saved models
        """
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.models = {}
        self.model_names = ['Logistic Regression', 'Naive Bayes', 'SVM']

        # Load models on initialization
        self.load_models()

    def load_models(self):
        """
        Load trained models and vectorizer from disk.

        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            # Load vectorizer
            vectorizer_path = f"{self.models_dir}/tfidf_vectorizer.pkl"
            self.vectorizer = joblib.load(vectorizer_path)

            # Load models
            for model_name in self.model_names:
                model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
                model_path = f"{self.models_dir}/{model_filename}"

                try:
                    self.models[model_name] = joblib.load(model_path)
                except FileNotFoundError:
                    print(f"Warning: {model_name} model not found at {model_path}")
                    continue

            if not self.models:
                print("Error: No models could be loaded")
                return False

            print(f"Successfully loaded {len(self.models)} models")
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def preprocess_text(self, text):
        """
        Preprocess a single text input.

        Args:
            text (str): Input text to preprocess

        Returns:
            str: Preprocessed text
        """
        return self.preprocessor.preprocess_text(text)

    def vectorize_text(self, text):
        """
        Convert preprocessed text to TF-IDF vectors.

        Args:
            text (str): Preprocessed text

        Returns:
            sparse matrix: TF-IDF vector representation
        """
        return self.vectorizer.transform([text])

    def predict_single(self, text, model_name='Logistic Regression'):
        """
        Make a prediction for a single text input.

        Args:
            text (str): Input text to classify
            model_name (str): Name of the model to use for prediction

        Returns:
            dict: Prediction results including label, confidence, and probabilities
        """
        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")

        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Vectorize text
        text_vector = self.vectorize_text(processed_text)

        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(text_vector)[0]

        # Get prediction probabilities/confidence
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vector)[0]
            confidence = max(probabilities)
            prob_fake = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            prob_real = probabilities[0]
        else:
            # For SVM without probability=True, use decision function
            decision_value = model.decision_function(text_vector)[0]
            # Convert decision value to probability-like score
            confidence = 1 / (1 + np.exp(-abs(decision_value)))
            prob_fake = confidence if decision_value > 0 else (1 - confidence)
            prob_real = 1 - prob_fake

        # Convert numeric prediction to label
        label = "Fake News" if prediction == 1 else "Real News"

        result = {
            'text': text,
            'processed_text': processed_text,
            'prediction': label,
            'prediction_numeric': int(prediction),
            'confidence': round(float(confidence), 4),
            'probability_real': round(float(prob_real), 4),
            'probability_fake': round(float(prob_fake), 4),
            'model_used': model_name
        }

        return result

    def predict_multiple_models(self, text):
        """
        Make predictions using all available models.

        Args:
            text (str): Input text to classify

        Returns:
            list: List of predictions from all models
        """
        results = []

        for model_name in self.model_names:  # Use model_names to maintain order
            if model_name not in self.models:
                continue
            try:
                result = self.predict_single(text, model_name)
                results.append(result)
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                continue

        return results

    def get_model_info(self):
        """
        Get information about loaded models.

        Returns:
            dict: Information about available models
        """
        return {
            'available_models': list(self.models.keys()),
            'vectorizer_info': {
                'max_features': self.vectorizer.max_features,
                'ngram_range': self.vectorizer.ngram_range,
                'vocabulary_size': len(self.vectorizer.vocabulary_)
            }
        }

    def predict_with_ensemble(self, text, method='majority_vote'):
        """
        Make ensemble prediction using multiple models.

        Args:
            text (str): Input text to classify
            method (str): Ensemble method ('majority_vote' or 'average_probability')

        Returns:
            dict: Ensemble prediction result
        """
        if not self.models:
            raise ValueError("No models available for ensemble prediction")

        # Get predictions from all models
        all_predictions = self.predict_multiple_models(text)

        if not all_predictions:
            raise ValueError("Could not get predictions from any model")

        predictions = []
        probabilities_fake = []

        for model_name, result in all_predictions.items():
            predictions.append(result['prediction_numeric'])
            probabilities_fake.append(result['probability_fake'])

        if method == 'majority_vote':
            # Majority vote
            ensemble_prediction = 1 if sum(predictions) > len(predictions) / 2 else 0
            confidence = sum(predictions) / len(predictions)

        elif method == 'average_probability':
            # Average probability
            avg_prob_fake = np.mean(probabilities_fake)
            ensemble_prediction = 1 if avg_prob_fake > 0.5 else 0
            confidence = max(avg_prob_fake, 1 - avg_prob_fake)

        else:
            raise ValueError("Invalid ensemble method. Use 'majority_vote' or 'average_probability'")

        label = "Fake News" if ensemble_prediction == 1 else "Real News"

        ensemble_result = {
            'text': text,
            'prediction': label,
            'prediction_numeric': ensemble_prediction,
            'confidence': round(float(confidence), 4),
            'method': method,
            'individual_predictions': all_predictions
        }

        return ensemble_result