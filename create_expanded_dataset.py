"""
Expanded dataset for fake news detection.
This dataset includes diverse real and fake news samples to improve model training.
"""

import pandas as pd
import os

# Create expanded dataset with more diverse and realistic samples
expanded_dataset = {
    'text': [
        # REAL NEWS SAMPLES (indicators: factual, organization names, specific numbers, verifiable)
        "Scientists have discovered a new species of deep-sea fish off the coast of Japan. The research team from the University of Tokyo published their findings in the Journal of Marine Biology.",
        "The Federal Reserve announced today that it will raise interest rates by 0.25 percent. This decision comes after months of debate among board members about inflation management.",
        "Stock markets around the world faced significant volatility today as investors responded to new economic data. The S&P 500 index fell 2.3 percent during trading.",
        "A new cure for a rare genetic disorder has been approved by the FDA. Clinical trials spanning five years showed a 95 percent success rate in patients.",
        "The World Health Organization reported that global vaccination rates have reached 70 percent of the population. This milestone was achieved through international cooperation.",
        "Apple Inc. announced record quarterly profits of 25.5 billion dollars, driven by strong iPhone sales. Revenue increased 12 percent compared to the same quarter last year.",
        "Researchers from Harvard University published a study correlating regular exercise with improved heart health. The longitudinal study followed 50,000 participants over ten years.",
        "The European Space Agency successfully launched a new satellite to monitor climate change. The satellite is equipped with advanced sensors to track atmospheric changes.",
        "A new treatment for Alzheimer's disease has shown promise in clinical trials. The medication managed to slow cognitive decline by 35 percent in patient groups.",
        "The Department of Transportation unveiled new infrastructure improvements totaling 100 billion dollars focused on road repairs and public transit systems.",
        "Researchers at Stanford developed a new battery technology that doubles energy capacity. The breakthrough was published in Nature Materials this week.",
        "NASA's Mars rover discovered evidence of ancient microbial life. The discovery was confirmed through analysis of rock samples collected over the past year.",
        "Google announced new AI capabilities that improve translation accuracy to 99.2 percent through improved neural network architecture.",
        "A team of biologists discovered a new mechanism for cellular regeneration that could lead to treatments for various diseases.",
        "IBM released a new quantum processor with 433 qubits representing significant advancement in quantum computing technology.",
        "Tesla reported record vehicle deliveries of 1.3 million units in the fiscal year, a 40 percent increase from the previous year.",
        "The International Labour Organization reported unemployment dropped to 3.5 percent globally, the lowest rate in decades.",
        "Microsoft announced a major investment in cloud infrastructure of 5 billion dollars to expand data centers across five continents.",
        "Oil prices fell 8 percent following news of increased production output with predictions for price stabilization.",
        "Amazon opened 100 new warehouses to expand delivery capabilities, creating 50,000 new jobs.",
        "The CDC published guidelines for a new preventive treatment for cardiovascular disease reducing heart attack risk by 30 percent.",
        "A multinational study of 100,000 participants shows that walking 30 minutes daily reduces mortality according to results published in The Lancet.",
        "Johns Hopkins University reports breakthrough in cancer immunotherapy showing 60 percent remission rates in clinical trials.",
        "Economists announced that consumer spending increased 3.2 percent in the latest quarter driven by strong employment growth.",
        "The United Nations released a report confirming successful poverty reduction in developing nations over the past decade.",
        
        # FAKE NEWS SAMPLES (indicators: sensationalism, unverified claims, conspiracy language, emotional manipulation)
        "SHOCKING: Top Secret Government Reveals Hidden Alien Base! Whistleblowers claim the government has harbored extraterrestrials for decades.",
        "BREAKING: Vaccines Contain Microchips for Mind Control! Scientists warn that billions have been unknowingly implanted with tracking devices.",
        "EXPOSED: Celebrity Caught in Illegal Activity! Anonymous sources claim a major Hollywood star has been involved in criminal activities.",
        "CONSPIRACY: 5G Networks Deliberately Spread Disease! Expert researchers reveal tower emissions cause mysterious illnesses.",
        "URGENT: Government Secret Plan to Control Population! Leaked documents show sinister agenda to manipulate global population growth.",
        "WARNING: Drinking Water Poisoned by Corporations! Whistleblower reveals massive cover-up of toxic contamination in all water supplies.",
        "EXCLUSIVE: Global Elites Planning New World Order! Anonymous sources reveal plans for total world domination and complete control.",
        "SHOCKING TRUTH: Moon Landing Was a Hoax! Scientists finally admit the truth about Apollo missions being filmed in studios.",
        "ALERT: Bill Gates Planning Mass Vaccination to Reduce Population! Philanthropist's real agenda exposed by insider sources.",
        "BREAKING NEWS: Royals Hidden For Decades! Shocking secrets about the royal family leaked by unnamed sources worldwide.",
        "EXCLUSIVE: Celebrity's Secret Child Revealed! Industry insiders expose shocking family secrets nobody knew about.",
        "SCANDAL: Government Officials Caught in Cover-Up! Documents prove major conspiracy hidden from public for years.",
        "SHOCKING: New World Order Proof Found! Leaked meetings show world leaders planning global takeover of all governments.",
        "EXPOSED: Major Corporation Hidden Crimes! Whistleblower reveals illegal activities that authorities are covering up.",
        "BREAKING: Scientists Hide Discovery That Changes Everything! Researchers suppress groundbreaking finding that contradicts official narrative.",
        "ALERT: Chemtrails Secretly Affecting Your Health! Former pilots confirm government spraying programs designed to control population.",
        "EXCLUSIVE: UFO Evidence US Government Hiding! Military documents prove aliens have visited Earth multiple times.",
        "SHOCKING: Vaccines Linked to Autism Epidemic! Doctors confirm connection despite official denials and cover-ups.",
        "URGENT: Financial Elite Planning Global Collapse! Insiders reveal plan to crash economy and establish one-world government.",
        "WARNING: Fluoride in Water is Population Control! Experts reveal government conspiracy to modify human behavior through water.",
        "BREAKING: Ancient Prophecies Predicting Current Events! Scholars reveal hidden messages predicting exactly what is happening today.",
        "EXPOSED: Hollywood Figures in Secret Society! Anonymous sources reveal shocking membership in mysterious organizations.",
        "SHOCKING: Technology Giant Creating Surveillance State! Whistleblower exposes plans for complete world monitoring system.",
        "EXCLUSIVE: Missing Evidence Proves Conspiracy! Suppressed reports confirm government involvement in major historical events.",
        "ALERT: Mysterious Illness Spreading Globally! Health organizations hiding true cause of widespread disease outbreak.",
    ],
    'label': [
        # Real news labels (25 total)
        'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL',
        'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL', 'REAL',
        'REAL', 'REAL', 'REAL', 'REAL', 'REAL',
        # Fake news labels (25 total - now balanced)
        'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE',
        'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE',
        'FAKE', 'FAKE', 'FAKE', 'FAKE', 'FAKE',
    ]
}

# Create DataFrame
df = pd.DataFrame(expanded_dataset)

# Save to CSV
output_path = 'dataset/expanded_news.csv'
os.makedirs('dataset', exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Expanded dataset created with {len(df)} samples")
print(f"   Real News: {(df['label'] == 'REAL').sum()} samples")
print(f"   Fake News: {(df['label'] == 'FAKE').sum()} samples")
print(f"   Saved to: {output_path}")
