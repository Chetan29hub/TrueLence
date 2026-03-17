"""
Enhanced dataset with more diverse and realistic news samples
"""

import pandas as pd
import os

# Create much larger and more diverse dataset
dataset = {
    'text': [
        # ===== REAL NEWS - Science/Technology (10 samples) =====
        "NASA's James Webb Space Telescope discovers water vapor on distant exoplanet. The telescope detected water signatures in the atmosphere of K2-18 b, a planet 120 light-years from Earth.",
        "Researchers at Stanford develop new battery technology that doubles energy capacity. The breakthrough was published in Nature Materials and could revolutionize electric vehicle range.",
        "Apple releases iPhone 15 Pro with advanced camera system featuring 48MP main sensor. Pre-orders begin September 15 with prices starting at $999 for the base model.",
        "Google announces Gemini AI model with improved reasoning capabilities. The company claims 40% improvement in mathematical problem-solving over previous models.",
        "Open AI releases GPT-4 with enhanced multimodal capabilities. The model can now process both text and images with improved accuracy and context understanding.",
        "Scientists discover new particle at CERN that may help explain dark matter. Physicists conducted 10 years of experiments to confirm the discovery.",
        "Intel announces new processor architecture breakthrough. The chip achieves 30% better performance per watt than previous generation.",
        "Microsoft invests 10 billion dollars in artificial intelligence research. The investment aims to advance AI capabilities across multiple applications.",
        "Amazon opens 100 new robotics-powered warehouses. The automation is expected to improve delivery speeds by 25 percent.",
        "Tesla reports record quarterly deliveries of 1.8 million vehicles. Revenue increased 15 percent year-over-year despite market challenges.",
        
        # ===== REAL NEWS - Health/Medicine (10 samples) =====
        "WHO reports 15% reduction in malaria deaths globally. Increased funding for treatment and prevention led to the significant improvement.",
        "FDA approves new Alzheimer's drug showing 35% disease progression slowdown. Clinical trials involved 3,000 patients over 18 months.",
        "Doctors successfully perform world's first pig-to-human heart transplant. The groundbreaking procedure took 8 hours and required special genetic modifications.",
        "New blood test can detect cancer six months before symptoms. Researchers published findings in the journal Cancer Research.",
        "COVID-19 vaccine effectiveness remains high at 88% against severe illness. Updated analysis includes data from 50 countries.",
        "Breakthrough in Parkinson's treatment shows 40% symptom improvement. The therapy uses gene transfer technology.",
        "Study shows Mediterranean diet reduces heart disease risk by 30%. Research published in The New England Journal of Medicine.",
        "Researchers develop new treatment for Type 2 diabetes with minimal side effects. Phase 3 trials exceed expectations.",
        "Johns Hopkins University reports advances in cancer immunotherapy. Survival rates improved by 45% in clinical trials.",
        "Sleep deprivation linked to weakened immune response. Study conducted by the CDC on 10,000 participants.",
        
        # ===== REAL NEWS - Business/Economics (10 samples) =====
        "Federal Reserve maintains interest rates at 5.25-5.50 percent. Central bank signals potential cuts later in the year.",
        "S&P 500 reaches all-time high of 5,400 points. Tech stocks lead market gains with 20% year-to-date performance.",
        "Nvidia becomes most valuable company with market cap exceeding 3 trillion dollars. Stock has risen 240% in the past year.",
        "JPMorgan Chase reports record quarterly profits of 15 billion dollars. CEO Jamie Dimon attributes growth to strong loan demand.",
        "Unemployment drops to 3.7% in the United States. Job market remains resilient with 500,000 positions added last month.",
        "Inflation slows to 3.2% annually. Federal Reserve announces confidence in economic stabilization.",
        "China GDP grows 5.2% in the quarter. Government credits manufacturing and export strength.",
        "Oil prices stabilize at 85 dollars per barrel. OPEC maintains current production levels.",
        "Housing starts increase 8% month-over-month. Real estate market shows signs of recovery.",
        "Cryptocurrency market recovers with Bitcoin trading at 42,000 dollars. Institutional investors increase positions.",
        
        # ===== FAKE NEWS - Conspiracy/Misinformation (10 samples) =====
        "SHOCKING: Top scientists reveal Earth is actually flat! NASA has been hiding the truth for 50 years. Photos distributed show edge of the world.",
        "ALERT: Bill Gates admits COVID vaccines contain microchips for mind control! Leaked audio reveals billionaire's plan for global population surveillance.",
        "BREAKING: 5G towers causing bird extinction! Dead birds found near cell towers. Scientists confirm electromagnetic radiation is wiping out species.",
        "EXCLUSIVE: Time travel device invented by secret government lab! DARPA successfully sent objects back in time. Whistleblower reveals plans.",
        "WARNING: Chemtrails are poisoning the population! Former pilots confirm government spraying program designed to control population behavior.",
        "PROOF: Alien base hidden under the White House! Underground facility holds extraterrestrial technology. Military officials confirmed the discovery.",
        "URGENT: New World Order plan exposed! Leaked Bilderberg meeting documents reveal conspiracy for world domination and one-world government.",
        "SHOCKING TRUTH: Moon landing was fake! NASA filmed the entire mission in a studio. Photos show shadows inconsistent with lunar conditions.",
        "BREAKING: Reptilians controlling world governments! Elite world leaders are actually shapeshifting alien reptiles conducting secret meetings.",
        "ALERT: Fluoride in water causes population control! Government adds mind-altering chemicals to drinking water to reduce fertility rates.",
        
        # ===== FAKE NEWS - Sensational/Clickbait (10 samples) =====
        "You won't believe what this celebrity's secret twin reveals! Hollywood star exposed for living double life for 20 years.",
        "SHOCKING: Billionaire found dead in mysterious circumstances! Officials claim accident but evidence suggests cover-up.",
        "This simple trick will make you 100 years old! Doctors hate her for revealing this one weird secret.",
        "EXPOSED: A-list actor's secret love child shocks entertainment world! Photos prove years-long hidden relationship.",
        "BREAKING: Royal family member arrested in stunning bust! Authorities reveal shocking criminal allegations involving multiple counts.",
        "This is why the government doesn't want you to know about! Classified information reveals truth authorities hide from public.",
        "SCANDAL: Famous politician caught in compromising situation! Anonymous photos emerge showing embezzlement and bribery.",
        "STUNNING: Celebrity couple's secret marriage revealed! They've been literally married in secret ceremony no one knew about.",
        "You won't believe the real reason for celebrity's sudden retirement! Inside sources reveal shocking health revelation.",
        "EXCLUSIVE: Top entertainer's double life exposed! Secret second family discovered in shocking scandal.",
        
        # ===== FAKE NEWS - Unverified Claims (10 samples) =====
        "New diet pill causes instant weight loss of 100 pounds! FDA-approved supplement melts fat while you sleep. Doctors shocked.",
        "Ancient pyramid discovered under Antarctic ice! 10,000-year-old structure found. Could rewrite human history.",
        "Miracle cure for all diseases discovered! Scientists develop pill that treats every known illness. Big Pharma suppressing findings.",
        "Lost city of Atlantis found! Underwater explorers discover advanced ancient civilization. Government keeping discovery secret.",
        "Historical records prove dinosaurs never existed! Fossils are elaborate hoax created by scientists. Real evidence hidden.",
        "New source of unlimited free energy discovered! Scientists harness quantum field to power entire civilizations.",
        "Immortality achieved by secret society! Wealthy elites have discovered fountain of youth and are hiding it from masses.",
        "Ancient aliens built all pyramids! Extraterrestrials constructed Egyptian monuments 50,000 years ago. Archaeologists falsify data.",
        "Hollow Earth theory confirmed! Scientists discover civilization inside the planet. NASA hiding truth from public.",
        "Bigfoot exists and runs government! Documented evidence shows sasquatch operates from underground facility.",
        
        # ===== BORDERLINE/EXAGGERATED (10 samples) =====
        "Study claims eating chocolate increases intelligence by significant amount. Harvard researchers find positive cognitive effects.",
        "New fitness trend promises rapid muscle development. Young athletes report noticeable gains within weeks of training.",
        "Celebrity reveals unconventional health practice with surprising benefits. Social media followers report improved energy levels.",
        "Investment opportunity offers unusually high returns. Financial advisor claims 25% guaranteed annual growth.",
        "Startup claims revolutionary technology will transform transportation industry. Early investors show interest in company.",
        "University research suggests link between lifestyle and longevity. Preliminary findings indicate correlation in study group.",
        "Local entrepreneur claims his invention could solve energy crisis. Prototype demonstration shows promising initial results.",
        "Wellness coach promotes alternative medicine approach gaining popularity. Enthusiasts report improved overall health outcomes.",
        "New social media platform claims to revolutionize communication. Beta testers express enthusiasm about platform features.",
        "Tech startup unveils controversial AI application. Developers claim algorithm provides accurate predictions.",
    ],
    'label': [
        # Real news labels
        'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL',  # Tech/Science
        'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL',  # Health/Medicine
        'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL',  # Business
        # Fake news labels
        'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE',  # Conspiracy
        'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE',  # Sensational
        'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE',  # Unverified
        # Borderline labels
        'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE',  # Exaggerated/Borderline
    ]
}

# Create DataFrame
df = pd.DataFrame(dataset)

# Save to CSV
output_path = 'dataset/news_training_data.csv'
os.makedirs('dataset', exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Enhanced training dataset created with {len(df)} samples")
print(f"   Real News: {(df['label'] == 'REAL').sum()} samples")
print(f"   Fake News: {(df['label'] == 'FAKE').sum()} samples")
print(f"   Saved to: {output_path}")
