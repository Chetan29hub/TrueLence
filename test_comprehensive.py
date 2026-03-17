"""
Fake News Detection - Test Script

This script tests the fake news detection system with various
real and fake news samples to demonstrate its capabilities.
"""

from src.prediction import FakeNewsPredictor

def main():
    """Test the fake news detection system with various samples."""

    print("📰 Fake News Detection - Comprehensive Testing")
    print("=" * 60)

    # Initialize predictor
    predictor = FakeNewsPredictor()

    # Test samples - Real News
    real_news_samples = [
        "Apple announces new iPhone 15 Pro with advanced camera system. The latest smartphone features a 48MP main camera, titanium build, and A17 Pro chip. Pre-orders begin September 15th with prices starting at $999.",
        "NASA's James Webb Space Telescope discovers water vapor on distant exoplanet. The telescope detected water signatures in the atmosphere of K2-18 b, a planet 120 light-years from Earth, raising hopes for potential habitability.",
        "Federal Reserve maintains interest rates at 5.25-5.50%. In a widely expected decision, the central bank kept borrowing costs steady while signaling potential cuts later this year amid cooling inflation.",
        "WHO reports decline in global malaria cases. New data shows a 7% reduction in malaria incidence worldwide in 2023, attributed to increased funding for mosquito nets and improved treatment access.",
        "Renewable energy hits record high in US. Solar and wind power accounted for 23% of U.S. electricity generation in Q1 2024, surpassing coal for the first time in history."
    ]

    # Test samples - Fake News
    fake_news_samples = [
        "SHOCKING EXPOSED: Bill Gates admits COVID vaccine contains microchip for mind control! In leaked audio, the billionaire reveals the true purpose of vaccines is global population tracking and control.",
        "BREAKING: Aliens land in New York City! Extraterrestrial spacecraft spotted hovering over Manhattan as world leaders prepare for first contact. Government cover-up underway!",
        "URGENT: 5G towers causing worldwide bird extinction! Thousands of dead birds found near cell towers. Scientists confirm electromagnetic radiation is wiping out entire species.",
        "EXCLUSIVE: Time travel device invented by secret government lab! DARPA successfully sends objects back in time. Whistleblower reveals plans to change historical events.",
        "PROOF: Flat Earth confirmed by NASA insider! Leaked documents show space agency has been hiding the truth about Earth's shape for decades. Globe model is biggest lie in history!"
    ]

    # Test samples - Borderline/Ambiguous
    borderline_samples = [
        "Study shows that eating chocolate increases intelligence by 200%. Harvard researchers discover that dark chocolate consumption leads to immediate IQ boost and enhanced problem-solving skills.",
        "New diet pill causes instant weight loss of 50 pounds! FDA-approved supplement melts fat while you sleep. Doctors shocked by unprecedented results in clinical trials.",
        "Ancient pyramid discovered under Antarctic ice! Explorers find 10,000-year-old structure that defies all known archaeological explanations. Could rewrite human history."
    ]

    print("\n🟢 TESTING REAL NEWS SAMPLES")
    print("-" * 40)

    real_correct = 0
    for i, text in enumerate(real_news_samples, 1):
        result = predictor.predict_single(text)
        prediction = result['prediction']
        confidence = result['confidence']

        is_correct = prediction == "Real News"
        if is_correct:
            real_correct += 1

        status = "✅" if is_correct else "❌"
        print(f"{status} Sample {i}: {prediction} ({confidence:.1%})")
        print(f"   Text: {text[:80]}...")

    print(f"\nReal News Accuracy: {real_correct}/{len(real_news_samples)} ({real_correct/len(real_news_samples)*100:.1f}%)")

    print("\n🔴 TESTING FAKE NEWS SAMPLES")
    print("-" * 40)

    fake_correct = 0
    for i, text in enumerate(fake_news_samples, 1):
        result = predictor.predict_single(text)
        prediction = result['prediction']
        confidence = result['confidence']

        is_correct = prediction == "Fake News"
        if is_correct:
            fake_correct += 1

        status = "✅" if is_correct else "❌"
        print(f"{status} Sample {i}: {prediction} ({confidence:.1%})")
        print(f"   Text: {text[:80]}...")

    print(f"\nFake News Accuracy: {fake_correct}/{len(fake_news_samples)} ({fake_correct/len(fake_news_samples)*100:.1f}%)")

    print("\n🟡 TESTING BORDERLINE SAMPLES")
    print("-" * 40)

    for i, text in enumerate(borderline_samples, 1):
        result = predictor.predict_single(text)
        prediction = result['prediction']
        confidence = result['confidence']

        print(f"⚠️  Sample {i}: {prediction} ({confidence:.1%})")
        print(f"   Text: {text[:80]}...")

    # Overall statistics
    total_correct = real_correct + fake_correct
    total_samples = len(real_news_samples) + len(fake_news_samples)
    overall_accuracy = total_correct / total_samples * 100

    print("\n📊 OVERALL PERFORMANCE")
    print("=" * 40)
    print(f"Total Samples Tested: {total_samples}")
    print(f"Correct Predictions: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    print(f"Real News Detection: {real_correct}/{len(real_news_samples)}")
    print(f"Fake News Detection: {fake_correct}/{len(fake_news_samples)}")

    print("\n🔄 MODEL COMPARISON ON SAMPLE 1")
    print("-" * 40)

    sample_text = real_news_samples[0]  # First real news sample
    comparison = predictor.predict_multiple_models(sample_text)

    for model_name, result in comparison.items():
        print(f"{model_name}: {result['prediction']} ({result['confidence']:.1%})")

if __name__ == "__main__":
    main()