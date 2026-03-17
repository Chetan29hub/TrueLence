"""
Fake News Detection - Demo Script

This script demonstrates how to use the trained models
for fake news detection without the web interface.
"""

from src.prediction import FakeNewsPredictor

def main():
    """Demo function showing how to use the prediction system."""

    print("📰 Fake News Detection Demo")
    print("=" * 40)

    # Initialize predictor
    predictor = FakeNewsPredictor()

    # Sample news articles for testing
    test_articles = [
        "NASA confirms discovery of Earth-like planet in habitable zone. Scientists believe Kepler-452b could support liquid water and potentially life.",
        "BREAKING: Aliens land in Washington DC! Government officials in secret meetings with extraterrestrial beings. Cover-up exposed!",
        "Federal Reserve reports strong economic growth with unemployment at historic lows. GDP expands by 3.2% in Q4.",
        "SHOCKING: Vaccines contain microchips for mind control! Bill Gates planning global population control through COVID vaccines."
    ]

    print("Testing with sample articles:\n")

    for i, article in enumerate(test_articles, 1):
        print(f"Article {i}:")
        print(f"Text: {article[:100]}...")

        # Make prediction
        result = predictor.predict_single(article)

        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Real Probability: {result['probability_real']:.1%}")
        print(f"Fake Probability: {result['probability_fake']:.1%}")
        print("-" * 50)

    # Demonstrate model comparison
    print("\n🤖 Model Comparison for Article 1:")
    comparison = predictor.predict_multiple_models(test_articles[0])

    for model_name, result in comparison.items():
        print(f"{model_name}: {result['prediction']} ({result['confidence']:.1%} confidence)")

if __name__ == "__main__":
    main()