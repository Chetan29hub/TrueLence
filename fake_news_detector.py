import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def clean_text(text):
    """
    Step 4: Clean the text data
    Removes punctuation, numbers, links, and converts text to lowercase.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower() # Convert text to lowercase
    # Remove the Reuters bias from Kaggle dataset (e.g. "WASHINGTON (Reuters) -")
    text = re.sub(r'^.*?\(reuters\)\s*-\s*', '', text)
    text = re.sub(r'reuters', '', text) 
    text = re.sub(r'\[.*?\]', '', text) # Remove text in square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', '', text) # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    return text

def main():
    print("========================================")
    print("      FAKE NEWS DETECTION AI BUILDER    ")
    print("========================================\n")

    # ----- 1. Load the dataset -----
    print("Step 1: Loading 'Fake.csv' and 'True.csv' from your 'archive' folder...")
    
    try:
        # Load the two separate files from the Kaggle dataset
        fake_df = pd.read_csv('archive/Fake.csv')
        true_df = pd.read_csv('archive/True.csv')
        
        # Merge 'title' and 'text' to give the model full context
        fake_df['text'] = fake_df['title'] + " " + fake_df['text']
        true_df['text'] = true_df['title'] + " " + true_df['text']
        
        # Add a custom 'label' column to both datasets
        fake_df['label'] = 0  # 0 indicates Fake
        true_df['label'] = 1  # 1 indicates True/Real
        
        # Inject your diverse expanded dataset!
        try:
            extra_df = pd.read_csv('dataset/expanded_news.csv')
            # Map the text labels to match the 0 and 1 format
            extra_df['label'] = extra_df['label'].map({'REAL': 1, 'FAKE': 0})
            df = pd.concat([fake_df, true_df, extra_df], ignore_index=True)
            print("Successfully injected 'expanded_news.csv' for a smarter AI!")
        except Exception as e:
            df = pd.concat([fake_df, true_df], ignore_index=True)
            
        # Shuffle the combined dataset randomly so True and Fake are mixed
        df = df.sample(frac=1).reset_index(drop=True)
        
    except FileNotFoundError:
        print("\n[ERROR] 'archive/Fake.csv' or 'archive/True.csv' not found!")
        print("Please make sure they are located inside the 'archive' folder inside your project directory.")
        return

    text_col = 'text'
    label_col = 'label'
    
    if text_col not in df.columns or label_col not in df.columns:
        print(f"\n[ERROR] Missing expected columns. Found columns: {list(df.columns)}")
        return

    # ----- 2. Handle missing values -----
    print("\nStep 2: Handling missing values...")
    original_size = len(df)
    df = df.dropna(subset=[text_col, label_col])  # Drop rows where text or label is NaN/Empty
    print(f"Dropped {original_size - len(df)} rows with missing data. Remaining rows: {len(df)}")

    # ----- 3. Show dataset columns and basic info -----
    print("\nStep 3: Dataset Columns & Info...")
    print(df.info())
    print(f"\nExample Labels found: {df[label_col].unique()}")

    # ----- 4. Clean the text data (Preprocessing) -----
    print("\nStep 4: Cleaning text data (lowercasing, removing punctuation...)")
    # Apply the clean_text function we generated above to the entire text column
    df['clean_text'] = df[text_col].apply(clean_text)

    # ----- 5 & 6. Split dataset & Convert into numerical form (TF-IDF) -----
    # Define features (X) and target/labels (y)
    X = df['clean_text']
    y = df[label_col]

    print("\nStep 5 & 6: Splitting the dataset & Vectorizing with TF-IDF...")
    # Split the dataset into 80% Training and 20% Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize TF-IDF Vectorizer to convert text to numerical form
    # We limit to 5000 max_features so the model trains quickly on normal machines
    vectorizer = TfidfVectorizer(max_features=5000)

    # FIT the vectorizer on the training data and transform it
    xv_train = vectorizer.fit_transform(X_train)
    # Only TRANSFORM the test data (never fit on test data!)
    xv_test = vectorizer.transform(X_test)

    # ----- 7. Train a machine learning model -----
    print("\nStep 7: Training Logistic Regression Model...")
    model = LogisticRegression()
    model.fit(xv_train, y_train)

    # ----- 8. Evaluate the model using accuracy score -----
    print("\nStep 8: Evaluating the model on unseen test data...")
    predictions = model.predict(xv_test)
    score = accuracy_score(y_test, predictions)
    print(f"--> Model Accuracy: {round(score * 100, 2)}%")

    # ----- 9. Allow manual input to test whether a news is Fake or Real -----
    print("\n========================================")
    print("   Step 9: MANUAL INPUT TESTING READY   ")
    print("========================================")
    print("Type your news paragraph below to test it.")
    print("Type 'exit' or 'quit' to stop the script.\n")

    while True:
        user_input = input("Enter news text to check: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("\nExiting the AI Detector. Goodbye!")
            break
            
        if not user_input.strip():
            print("Please enter a valid string.")
            continue
            
        # Clean the manual user input using our cleaning function
        clean_input = clean_text(user_input)
        
        # Convert the cleaned text to numerical format using the fitted TF-IDF
        vectorized_input = vectorizer.transform([clean_input])
        
        # Make Prediction
        prediction = model.predict(vectorized_input)
        predicted_label = prediction[0]
        
        # Print the output result
        # Note: You may need to swap the visual printing if your CSV maps "Fake" to 0 or 1 differently.
        # Generally: If your label string contains "Fake" (case-insensitive) or is mapped to an int.
        if isinstance(predicted_label, str):
            if "fake" in predicted_label.lower():
                print(f"Prediction: ---> 🔴 {predicted_label.upper()} NEWS")
            else:
                print(f"Prediction: ---> 🟢 {predicted_label.upper()} NEWS")
        else:
            # Assuming 0 is Fake and 1 is Real as typical configurations
            if predicted_label == 0:
                 print("Prediction: ---> 🔴 FAKE NEWS (Class 0)")
            else:
                 print("Prediction: ---> 🟢 REAL NEWS (Class 1)")
                 
        print("-" * 50)

if __name__ == "__main__":
    main()
