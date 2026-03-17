"""
Fake News Detection - Model Training Module

This module contains functions for training machine learning models
for fake news detection using various algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
from .preprocessing import TextPreprocessor

class FakeNewsModel:
    """
    A class for training and evaluating fake news detection models.
    """

    def __init__(self):
        """
        Initialize the FakeNewsModel with preprocessing and ML components.
        """
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.models = {}
        # All models are classifiers (not regression)
        self.model_names = ['Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forest']

    def load_data(self, file_path):
        """
        Load and prepare the dataset.

        Args:
            file_path (str): Path to the CSV file containing news data

        Returns:
            tuple: (X, y) where X is text data and y is labels
        """
        try:
            df = pd.read_csv(file_path)

            # Check for different possible column names
            text_columns = ['text', 'content', 'article', 'news']
            label_columns = ['label', 'target', 'class', 'fake']

            text_col = None
            label_col = None

            for col in text_columns:
                if col in df.columns:
                    text_col = col
                    break

            for col in label_columns:
                if col in df.columns:
                    label_col = col
                    break

            if text_col is None or label_col is None:
                raise ValueError("Could not find appropriate text or label columns in the dataset")

            X = df[text_col].fillna('')
            y = df[label_col]

            # Convert labels to binary (0 for real, 1 for fake)
            if y.dtype == 'object':
                y = y.map({'REAL': 0, 'FAKE': 1, 'real': 0, 'fake': 1})
            else:
                # Assume 0 is real, 1 is fake
                y = y.astype(int)

            print(f"Dataset loaded successfully. Shape: {df.shape}")
            print(f"Real news: {sum(y == 0)}, Fake news: {sum(y == 1)}")

            return X, y

        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def preprocess_and_vectorize(self, X_train, X_test):
        """
        Preprocess text data and convert to TF-IDF vectors.

        Args:
            X_train (pd.Series): Training text data
            X_test (pd.Series): Testing text data

        Returns:
            tuple: (X_train_vec, X_test_vec) vectorized data
        """
        print("Preprocessing text data...")

        # Preprocess training data
        X_train_processed = self.preprocessor.preprocess_batch(X_train.tolist())

        # Preprocess testing data
        X_test_processed = self.preprocessor.preprocess_batch(X_test.tolist())

        print("Converting text to TF-IDF vectors...")

        # Initialize TF-IDF vectorizer with improved parameters
        # Tuned for fake news detection with balanced features
        self.vectorizer = TfidfVectorizer(
            max_features=2000,           # Number of features to extract
            ngram_range=(1, 3),          # Use unigrams, bigrams, and trigrams
            min_df=1,                    # Minimum document frequency (1 for small datasets)
            max_df=1.0,                  # Maximum document frequency
            lowercase=True,              # Ensure all text is lowercase
            stop_words=None,             # We've already handled stopwords in preprocessing
            sublinear_tf=True,           # Apply sublinear TF scaling
            use_idf=True,                # Enable IDF weighting
            smooth_idf=True,             # Prevent zero divisions
            norm='l2',                   # L2 normalization
            strip_accents='unicode',     # Remove accents
            analyzer='word',             # Use word analysis
            token_pattern=r'(?u)\b\w\w+\b',  # Match words with at least 2 characters
            dtype=np.float32             # Use 32-bit floats for memory efficiency
        )

        # Fit on training data and transform both sets
        X_train_vec = self.vectorizer.fit_transform(X_train_processed)
        X_test_vec = self.vectorizer.transform(X_test_processed)

        print(f"TF-IDF Vectorizer Info:")
        print(f"   Max features: 2000")
        print(f"   N-gram range: (1, 3)")
        print(f"   Features extracted: {X_train_vec.shape[1]}")
        print(f"   Training shape: {X_train_vec.shape}")
        print(f"   Test shape: {X_test_vec.shape}")

        return X_train_vec, X_test_vec

    def train_models(self, X_train_vec, y_train):
        """
        Train multiple ML models with tuned hyperparameters.

        Args:
            X_train_vec (sparse matrix): Vectorized training data
            y_train (pd.Series): Training labels
        """
        print("Training models with improved hyperparameters...")

        # Logistic Regression - Best for text classification with probability calibration
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear',           # Good for small datasets
            C=1.0,                        # Regularization strength
            class_weight='balanced',      # Handle class imbalance
            penalty='l2'                  # L2 regularization
        )
        lr_model.fit(X_train_vec, y_train)
        self.models['Logistic Regression'] = lr_model

        # Naive Bayes - Good baseline for text classification
        print("Training Naive Bayes...")
        nb_model = MultinomialNB(
            alpha=1.0,                    # Laplace smoothing parameter
            fit_prior=True,               # Learn class prior probabilities
            class_prior=None              # Use the empirical class distribution
        )
        nb_model.fit(X_train_vec, y_train)
        self.models['Naive Bayes'] = nb_model

        # Support Vector Machine - Strong classifier but prone to overfitting on small datasets
        print("Training SVM...")
        svm_model = SVC(
            kernel='linear',
            probability=True,
            random_state=42,
            C=1.0,                        # Regularization parameter
            class_weight='balanced',      # Handle class imbalance
            gamma='scale',                # Kernel coefficient
            max_iter=2000
        )
        svm_model.fit(X_train_vec, y_train)
        self.models['SVM'] = svm_model

        # Random Forest Classifier - tree-based ensemble classifier
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            max_depth=20,
            n_jobs=-1
        )
        rf_model.fit(X_train_vec, y_train)
        self.models['Random Forest'] = rf_model

        print("✓ All models trained successfully!")

    def evaluate_model(self, model, X_test_vec, y_test, model_name):
        """
        Evaluate a single model and return metrics.

        Args:
            model: Trained model
            X_test_vec (sparse matrix): Vectorized test data
            y_test (pd.Series): Test labels
            model_name (str): Name of the model

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test_vec)

        # Calculate probabilities for confidence scores
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_vec)[:, 1]
        else:
            y_prob = model.decision_function(X_test_vec)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)

        metrics = {
            'Model': model_name,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1 Score': round(f1, 4),
            'Confusion Matrix': conf_matrix.tolist()
        }

        return metrics

    def evaluate_all_models(self, X_test_vec, y_test):
        """
        Evaluate all trained models.

        Args:
            X_test_vec (sparse matrix): Vectorized test data
            y_test (pd.Series): Test labels

        Returns:
            list: List of evaluation results for each model
        """
        results = []

        for model_name in self.model_names:
            if model_name in self.models:
                metrics = self.evaluate_model(
                    self.models[model_name],
                    X_test_vec,
                    y_test,
                    model_name
                )
                results.append(metrics)

        return results

    def save_models(self, models_dir='models'):
        """
        Save trained models and vectorizer to disk.

        Args:
            models_dir (str): Directory to save models
        """
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Save vectorizer
        joblib.dump(self.vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))

        # Save each model
        for model_name, model in self.models.items():
            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, os.path.join(models_dir, filename))

        print(f"Models saved to {models_dir}/")

    def load_models(self, models_dir='models'):
        """
        Load trained models and vectorizer from disk.

        Args:
            models_dir (str): Directory containing saved models
        """
        try:
            # Load vectorizer
            self.vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))

            # Load models
            for model_name in self.model_names:
                filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
                filepath = os.path.join(models_dir, filename)
                if os.path.exists(filepath):
                    self.models[model_name] = joblib.load(filepath)

            print("Models loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def train_and_evaluate(self, data_path, test_size=0.2, random_state=42):
        """
        Complete pipeline: load data, preprocess, train models, evaluate.

        Args:
            data_path (str): Path to the dataset CSV file
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility

        Returns:
            list: Evaluation results for all models
        """
        # Load data
        X, y = self.load_data(data_path)
        if X is None:
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")

        # Preprocess and vectorize
        X_train_vec, X_test_vec = self.preprocess_and_vectorize(X_train, X_test)

        # Train models
        self.train_models(X_train_vec, y_train)

        # Evaluate models
        results = self.evaluate_all_models(X_test_vec, y_test)

        # Save models
        self.save_models()

        return results