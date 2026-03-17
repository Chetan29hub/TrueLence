"""
Quick test script to check a specific news sample
"""

from src.prediction import FakeNewsPredictor

# Test sample
test_text = "NASA's James Webb Space Telescope discovers water vapor on distant exoplanet. The telescope detected water signatures in the atmosphere of K2-18 b, a planet 120 light-years from Earth, raising hopes for potential habitability."

print("\n" + "="*80)
print("🔬 TESTING SPECIFIC NEWS SAMPLE")
print("="*80)

print(f"\n📰 Text:\n{test_text}\n")

# Initialize predictor
predictor = FakeNewsPredictor()

# Test with all models
results = predictor.predict_multiple_models(test_text)

print("📊 MODEL PREDICTIONS:\n")

for result in results:
    print(f"🤖 {result['model_used']}")
    print(f"   Prediction: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.4f}")
    print(f"   Real News: {result['probability_real']*100:.1f}%")
    print(f"   Fake News: {result['probability_fake']*100:.1f}%")
    print()

# Consensus
predictions = [r['prediction'] for r in results]
real_count = predictions.count('Real News')
fake_count = predictions.count('Fake News')

print("="*80)
print(f"✅ CONSENSUS: {real_count}/3 models predict 'Real News' | {fake_count}/3 predict 'Fake News'")
print("="*80 + "\n")
