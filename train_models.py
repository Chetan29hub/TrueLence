"""
Fake News Detection - Training Script

This script demonstrates how to train the fake news detection models
using the training module. Run this script to train models on your dataset.
"""

import os
import sys
from src.model_training import FakeNewsModel

def main():
    """Main training function."""

    print("📰 Fake News Detection - Model Training")
    print("=" * 50)

    # Check if dataset exists
    dataset_path = 'dataset/news.csv'

    # If news.csv doesn't exist, try sample_news.csv
    if not os.path.exists(dataset_path):
        sample_path = 'dataset/sample_news.csv'
        if os.path.exists(sample_path):
            print(f"ℹ️  Using sample dataset: {sample_path}")
            dataset_path = sample_path
        else:
            print(f"❌ Dataset not found at: {dataset_path}")
            print("\n📝 Please ensure your dataset is placed in the dataset/ directory")
            print("   Expected format: CSV file with 'text' and 'label' columns")
            print("   Labels should be 'REAL'/'FAKE' or 0/1")
            print("\n💡 You can download sample datasets from:")
            print("   - Kaggle Fake News Dataset: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset")
            print("   - LIAR Dataset: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
            return

    try:
        # Initialize the model trainer
        print("🔧 Initializing model trainer...")
        trainer = FakeNewsModel()

        # Train and evaluate models
        print("🚀 Starting model training and evaluation...")
        print("   This may take a few minutes depending on dataset size...")

        results = trainer.train_and_evaluate(dataset_path)

        if results is None:
            print("❌ Training failed. Please check your dataset format.")
            return

        # Display results
        print("\n📊 Training Results")
        print("=" * 50)

        for result in results:
            print(f"\n🤖 {result['Model']}")
            print("-" * 30)
            print(f"   Accuracy:  {result['Accuracy']:.4f}")
            print(f"   Precision: {result['Precision']:.4f}")
            print(f"   Recall:    {result['Recall']:.4f}")
            print(f"   F1 Score:  {result['F1 Score']:.4f}")
            print(f"   Confusion Matrix:")
            print(f"     {result['Confusion Matrix']}")

        print("\n✅ Training completed successfully!")
        print("📁 Models saved to 'models/' directory")
        print("🌐 You can now run the web application with: streamlit run app.py")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("\n🔍 Troubleshooting tips:")
        print("   1. Check that your dataset has the correct format")
        print("   2. Ensure all required packages are installed: pip install -r requirements.txt")
        print("   3. Make sure NLTK data is downloaded (see README.md)")
        return

if __name__ == "__main__":
    main()