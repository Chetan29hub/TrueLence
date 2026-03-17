"""
Diagnostic script to analyze model bias and dataset issues.
"""

import pandas as pd
import numpy as np
from src.model_training import FakeNewsModel
from src.prediction import FakeNewsPredictor
import os

def diagnose_dataset():
    """Analyze dataset balance and characteristics."""
    print("\n" + "="*60)
    print("📊 DATASET DIAGNOSIS")
    print("="*60)
    
    dataset_path = 'dataset/sample_news.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return None
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"\n📈 Dataset Size: {len(df)} samples")
    print(f"📋 Columns: {df.columns.tolist()}")
    
    # Check label distribution
    label_dist = df['label'].value_counts()
    print(f"\n📊 Label Distribution:")
    for label, count in label_dist.items():
        pct = (count / len(df)) * 100
        print(f"   {label}: {count} samples ({pct:.1f}%)")
    
    # Calculate imbalance ratio
    if len(label_dist) > 1:
        ratio = label_dist.iloc[0] / label_dist.iloc[1]
        print(f"   Imbalance Ratio: {ratio:.2f}:1")
    
    # Text statistics
    print(f"\n📝 Text Statistics:")
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"   Average text length: {df['text_length'].mean():.0f} chars")
    print(f"   Average word count: {df['word_count'].mean():.0f} words")
    print(f"   Min words: {df['word_count'].min()}, Max words: {df['word_count'].max()}")
    
    return df

def test_preprocessing():
    """Test preprocessing pipeline."""
    print("\n" + "="*60)
    print("🔄 PREPROCESSING DIAGNOSIS")
    print("="*60)
    
    from src.preprocessing import TextPreprocessor
    
    preprocessor = TextPreprocessor()
    
    # Test sample texts
    test_texts = {
        "Real News": "Scientists discover new planet capable of supporting life. NASA confirms the discovery.",
        "Fake News": "SHOCKING: Government hiding alien invasion! Sources claim the White House covers up UFO sightings.",
        "Short Real": "Stock market reaches all-time high.",
        "Short Fake": "CONSPIRACY: 5G towers causing bird deaths!"
    }
    
    print("\n📝 Preprocessing Examples:")
    for label, text in test_texts.items():
        processed = preprocessor.preprocess_text(text)
        print(f"\n{label}:")
        print(f"   Original ({len(text)} chars): {text[:80]}...")
        print(f"   Processed ({len(processed)} chars): {processed[:80]}...")

def diagnose_current_models():
    """Test current model predictions."""
    print("\n" + "="*60)
    print("🤖 CURRENT MODEL DIAGNOSIS")
    print("="*60)
    
    if not os.path.exists('models'):
        print("❌ No trained models found. Run python train_models.py first")
        return
    
    try:
        predictor = FakeNewsPredictor()
        
        # Test samples
        test_samples = {
            "Real News": "Scientists discover new planet capable of supporting life. NASA confirms the discovery of Kepler-452b.",
            "Fake News": "SHOCKING: Government hiding alien invasion! Sources claim the White House has been covering up UFO sightings.",
            "Ambiguous": "New study shows benefits of daily exercise.",
            "Clickbait": "You won't believe what happens next! Celebrity reveals shocking secret!"
        }
        
        print("\n🧪 Model Predictions:")
        for label, text in test_samples.items():
            try:
                result = predictor.predict_single(text)
                print(f"\n{label}:")
                print(f"   Text: {text[:70]}...")
                print(f"   Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']:.4f}")
                print(f"   Real News: {result['probability_real']:.4f}")
                print(f"   Fake News: {result['probability_fake']:.4f}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test with multiple models
        print("\n\n🔀 Multi-Model Predictions:")
        result = predictor.predict_multiple_models(test_samples["Fake News"])
        for pred in result:
            print(f"\n{pred['model_used']}:")
            print(f"   Prediction: {pred['prediction']}")
            print(f"   Confidence: {pred['confidence']:.4f}")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")

def train_and_evaluate_with_diagnostics():
    """Train models with detailed diagnostics."""
    print("\n" + "="*60)
    print("🚀 TRAINING WITH DIAGNOSTICS")
    print("="*60)
    
    dataset_path = 'dataset/sample_news.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    trainer = FakeNewsModel()
    
    # Load and analyze data
    X, y = trainer.load_data(dataset_path)
    if X is None:
        return
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n✓ Class distribution in original data:")
    print(f"   Real News (0): {counts[0]}")
    print(f"   Fake News (1): {counts[1]}")
    
    # Train and evaluate
    results = trainer.train_and_evaluate(dataset_path)
    
    if results:
        print(f"\n✅ Training complete!")
        print(f"\n📊 Evaluation Metrics:")
        for result in results:
            print(f"\n{result['Model']}:")
            print(f"   Accuracy:  {result['Accuracy']:.4f}")
            print(f"   Precision: {result['Precision']:.4f}")
            print(f"   Recall:    {result['Recall']:.4f}")
            print(f"   F1 Score:  {result['F1 Score']:.4f}")
            print(f"   Confusion Matrix: {result['Confusion Matrix']}")

if __name__ == "__main__":
    print("\n🔍 FAKE NEWS DETECTOR - COMPLETE DIAGNOSIS")
    print("=" * 60)
    
    # Run diagnostics
    diagnose_dataset()
    test_preprocessing()
    diagnose_current_models()
    
    print("\n" + "="*60)
    print("✅ DIAGNOSIS COMPLETE")
    print("="*60)
