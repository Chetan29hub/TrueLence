import pandas as pd
import os
import kagglehub
from src.model_training import FakeNewsModel

def main():
    print("="*70)
    print("📥 DOWNLOADING KAGGLE DATASET")
    print("="*70)
    
    # Download the dataset using kagglehub
    try:
        print("Downloading from Kaggle (this might take a few moments for 40MB+)...")
        path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
        print("✅ Download complete! Path to dataset files:", path)
    except Exception as e:
        print(f"❌ Failed to download dataset using kagglehub: {e}")
        return
        
    print("\n" + "="*70)
    print("🔄 PREPARING DATASET")
    print("="*70)
    
    try:
        # Load the True and Fake datasets
        true_df = pd.read_csv(os.path.join(path, "True.csv"))
        fake_df = pd.read_csv(os.path.join(path, "Fake.csv"))
        
        # Add labels based on which file they came from
        true_df['label'] = 'REAL' 
        fake_df['label'] = 'FAKE'
        
        # Combine the datasets
        combined_df = pd.concat([true_df, fake_df], ignore_index=True)
        
        print(f"Total Combined articles: {len(combined_df)}")
        
        dataset_path = 'dataset/kaggle_combined_news.csv'
        os.makedirs('dataset', exist_ok=True)
        
        # NOTE: This dataset has ~44,000 articles. Training SVM/RandomForest on 44,000 
        # text files can take a VERY long time (15+ minutes) on a standard PC.
        # To make it train in a reasonable time, we'll take a random sample of 10,000 rows.
        # This is still 200x larger than your original 50 row dataset!
        
        num_samples = min(10000, len(combined_df))
        sampled_df = combined_df.sample(n=num_samples, random_state=42)
        
        sampled_df.to_csv(dataset_path, index=False)
        print(f"Created a dataset sample at {dataset_path} with {len(sampled_df)} articles to ensure training finishes quickly.")
        
    except Exception as e:
        print(f"❌ Error preparing Kaggle data: {e}")
        return
        
    print("\n" + "="*70)
    print("🚀 TRAINING MODEL ON LARGE KAGGLE DATASET")
    print("="*70)
    
    try:
        trainer = FakeNewsModel()
        
        # Train on the new dataset using an 80/20 train/test split
        results = trainer.train_and_evaluate(dataset_path, test_size=0.2, random_state=42)
        
        if results is None:
            print("❌ Training failed.")
            return

        print("\n" + "="*70)
        print("📊 COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*70)

        for result in results:
            print(f"\n🤖 {result['Model']}")
            print("-" * 70)
            print(f"   Accuracy:  {result['Accuracy']:.4f}")
            print(f"   Precision: {result['Precision']:.4f}")
            print(f"   Recall:    {result['Recall']:.4f}")
            print(f"   F1 Score:  {result['F1 Score']:.4f}")
            
        print("\n✅ New smarter models have been successfully trained and saved to the 'models/' folder.")
        print("   Restart your application (python app_flask.py) to use them!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
