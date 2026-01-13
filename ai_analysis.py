import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🤖 AI-POWERED DATA ANALYSIS PROJECT")
print("="*70)

# ============================================================================
# STEP 1: GENERATE SAMPLE DATA
# ============================================================================
print("\n📊 Generating sample data...")

# Sales data
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
np.random.seed(42)

sales_df = pd.DataFrame({
    'date': dates,
    'revenue': np.random.normal(5000, 1000, len(dates)),
    'orders': np.random.randint(40, 70, len(dates)),
    'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates))
})

# Customer reviews
reviews = [
    "Great product! Fast delivery and excellent quality.",
    "Disappointed with customer service response time.",
    "Product arrived damaged but refund was quick.",
    "Amazing features, exactly what I needed for work.",
    "Price is too high compared to competitors.",
    "Excellent value for money, highly recommend!",
    "Product quality is poor, not worth the price.",
    "Customer support was very helpful and responsive.",
    "Shipping took forever but product is decent.",
    "Best purchase I've made this year!",
]

reviews_df = pd.DataFrame({
    'review': reviews,
    'date': pd.date_range('2024-01-01', periods=10, freq='10D')
})

print(f"✅ Created {len(sales_df)} sales records and {len(reviews_df)} reviews")

# ============================================================================
# STEP 2: AI SENTIMENT ANALYSIS
# ============================================================================
print("\n🤖 Loading AI model for sentiment analysis...")
sentiment_analyzer = pipeline("sentiment-analysis",
                             model="distilbert-base-uncased-finetuned-sst-2-english")

print("📝 Analyzing customer sentiment...")
sentiments = []
scores = []

for review in reviews_df['review']:
    result = sentiment_analyzer(review)[0]
    sentiments.append(result['label'])
    scores.append(result['score'])

reviews_df['sentiment'] = sentiments
reviews_df['confidence'] = scores

positive = sum(1 for s in sentiments if s == 'POSITIVE')
negative = sum(1 for s in sentiments if s == 'NEGATIVE')

print(f"\n✨ SENTIMENT ANALYSIS RESULTS:")
print(f"   Positive: {positive} ({positive/len(sentiments)*100:.0f}%)")
print(f"   Negative: {negative} ({negative/len(sentiments)*100:.0f}%)")
print(f"   Avg Confidence: {np.mean(scores):.1%}")

# ============================================================================
# STEP 3: ANOMALY DETECTION
# ============================================================================
print("\n🔍 Detecting anomalies in sales data...")

sales_df['rolling_mean'] = sales_df['revenue'].rolling(7).mean()
sales_df['rolling_std'] = sales_df['revenue'].rolling(7).std()
sales_df['z_score'] = (sales_df['revenue'] - sales_df['rolling_mean']) / sales_df['rolling_std']
sales_df['is_anomaly'] = abs(sales_df['z_score']) > 3

anomalies = sales_df[sales_df['is_anomaly']]
print(f"⚠️  Found {len(anomalies)} anomalies in revenue data")

# ============================================================================
# STEP 4: CREATE VISUALIZATIONS
# ============================================================================
print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('AI-Powered Data Analysis Dashboard', fontsize=16, fontweight='bold')

# 1. Revenue Trend
axes[0, 0].plot(sales_df['date'], sales_df['revenue'], alpha=0.6, linewidth=1)
axes[0, 0].plot(sales_df['date'], sales_df['rolling_mean'], 'r-', linewidth=2, label='7-day avg')
axes[0, 0].scatter(anomalies['date'], anomalies['revenue'], color='red', s=100,
                  marker='X', label='Anomalies', zorder=5)
axes[0, 0].set_title('Revenue Trend with Anomaly Detection', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Revenue ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Sentiment Analysis
sentiment_counts = reviews_df['sentiment'].value_counts()
colors = ['#2ecc71' if x == 'POSITIVE' else '#e74c3c' for x in sentiment_counts.index]
axes[0, 1].bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7)
axes[0, 1].set_title('AI Sentiment Analysis Results', fontweight='bold')
axes[0, 1].set_ylabel('Number of Reviews')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Regional Performance
regional = sales_df.groupby('region')['revenue'].sum().sort_values()
axes[1, 0].barh(regional.index, regional.values, color='skyblue')
axes[1, 0].set_title('Revenue by Region', fontweight='bold')
axes[1, 0].set_xlabel('Total Revenue ($)')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. Orders Distribution
axes[1, 1].hist(sales_df['orders'], bins=20, color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Daily Orders Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Number of Orders')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ai_analysis_results.png', dpi=300, bbox_inches='tight')
print("✅ Dashboard saved as 'ai_analysis_results.png'")

# ============================================================================
# STEP 5: GENERATE SUMMARY REPORT
# ============================================================================
print("\n" + "="*70)
print("📊 EXECUTIVE SUMMARY")
print("="*70)

total_revenue = sales_df['revenue'].sum()
avg_revenue = sales_df['revenue'].mean()
total_orders = sales_df['orders'].sum()

print(f"\n💰 FINANCIAL METRICS:")
print(f"   Total Revenue: ${total_revenue:,.0f}")
print(f"   Average Daily Revenue: ${avg_revenue:,.0f}")
print(f"   Total Orders: {total_orders:,}")
print(f"   Average Order Value: ${total_revenue/total_orders:.2f}")

print(f"\n🎭 CUSTOMER SENTIMENT (AI-Powered):")
print(f"   Customer Satisfaction: {positive/len(sentiments)*100:.0f}% Positive")
print(f"   Areas of Concern: {negative} negative reviews detected")

print(f"\n🔍 DATA QUALITY:")
print(f"   Anomalies Detected: {len(anomalies)} unusual patterns")
print(f"   Data Completeness: 100%")

print(f"\n🌍 TOP PERFORMING REGION:")
print(f"   {regional.index[-1]}: ${regional.values[-1]:,.0f}")

print("\n" + "="*70)
print("✨ Analysis complete! Check 'ai_analysis_results.png' for visuals")
print("="*70)

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================
print("\n💾 Saving results to CSV files...")

sales_df.to_csv('sales_analysis.csv', index=False)
reviews_df.to_csv('sentiment_analysis.csv', index=False)

print("✅ Saved: sales_analysis.csv")
print("✅ Saved: sentiment_analysis.csv")

