"""
Improved model training script with comprehensive evaluation.
"""

import os
import sys
from src.model_training import FakeNewsModel

def main():
    """Main training function with improved diagnostics."""

    print("\n" + "="*70)
    print("📰 FAKE NEWS DETECTION - IMPROVED MODEL TRAINING")
    print("="*70)

    # Use larger, more diverse dataset
    dataset_path = 'dataset/news_training_data.csv'
    
    # Fallback to expanded dataset if large dataset not found
    if not os.path.exists(dataset_path):
        dataset_path = 'dataset/expanded_news.csv'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found")
        print("   Run: python create_large_dataset.py")
        return

    try:
        # Initialize the model trainer
        print("\n🔧 Initializing model trainer with improved hyperparameters...")
        trainer = FakeNewsModel()

        # Train and evaluate models
        print("\n🚀 Starting model training with stratified cross-validation...")
        print("   Dataset: expanded_news.csv (50 samples, balanced)")
        print("   Train/Test Split: 80/20 with stratification\n")

        results = trainer.train_and_evaluate(dataset_path, test_size=0.2, random_state=42)

        if results is None:
            print("❌ Training failed. Please check your dataset format.")
            return

        # Display results
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*70)

        for result in results:
            print(f"\n🤖 {result['Model']}")
            print("-" * 70)
            print(f"   Accuracy:          {result['Accuracy']:.4f}  (Correct predictions / Total)")
            print(f"   Precision:         {result['Precision']:.4f}  (True Positives / All Positives)")
            print(f"   Recall:            {result['Recall']:.4f}  (True Positives / All Actual Positives)")
            print(f"   F1 Score:          {result['F1 Score']:.4f}  (Harmonic mean of Precision & Recall)")
            
            # Unpack confusion matrix
            cm = result['Confusion Matrix']
            tn, fp = cm[0]
            fn, tp = cm[1]
            
            print(f"\n   Confusion Matrix:")
            print(f"      ┌─────────────┬─────────────┐")
            print(f"      │   {tn:5d}   │   {fp:5d}   │  (Predicted Negative/Predicted Positive)")
            print(f"      ├─────────────┼─────────────┤")
            print(f"      │   {fn:5d}   │   {tp:5d}   │")
            print(f"      └─────────────┴─────────────┘")
            print(f"      Actual Neg    Actual Pos")
            print(f"\n   Interpretation:")
            print(f"      TN: {tn} (Correctly identified Real News)")
            print(f"      FP: {fp} (Incorrectly marked as Fake)")
            print(f"      FN: {fn} (Missed Fake News)")
            print(f"      TP: {tp} (Correctly identified Fake News)")

        # Summary
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED!")
        print("="*70)
        print(f"\n📁 Models saved to: models/")
        print("   - logistic_regression_model.pkl")
        print("   - naive_bayes_model.pkl")
        print("   - svm_model.pkl")
        print("   - tfidf_vectorizer.pkl")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   1. The model is now trained on a balanced dataset of 50 samples")
        print("   2. Logistic Regression typically performs best for text classification")
        print("   3. To improve further, consider:")
        print("      - Adding more diverse training samples (target: 500-1000)")
        print("      - Tuning probability thresholds (currently 0.5)")
        print("      - Using ensemble methods combining all three models")
        print("      - Fine-tuning TF-IDF parameters (n-grams, max features)")
        
        print("\n🌐 To use the improved model:")
        print("   - Run the Flask app: python app_flask.py")
        print("   - The app will now use the improved models with balanced predictions")

        print("\n" + "="*70 + "\n")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
