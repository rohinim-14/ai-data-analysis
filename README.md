# AI-Enhanced Data Analysis Platform

A learning project demonstrating the integration of modern AI/ML tools (Hugging Face) into real-world data analysis workflows.

## Project Overview

- **Sentiment Analysis** - Analyze customer feedback using NLP models
- **Topic Classification** - Categorize reviews with zero-shot learning
- **Anomaly Detection** - Identify unusual patterns in business metrics
- **Automated Insights** - Generate executive summaries using AI
- **Data Visualization** - Create comprehensive analytical dashboards

## Technologies Used

- **Python 3.8+**
- **Hugging Face Transformers** - Pre-trained AI models
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **PyTorch** - Deep learning backend

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-data-analysis.git
cd ai-data-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

##  Usage

Run the complete analysis pipeline:

```bash
python ai_data_analyst.py
```

This will:
1. Generate sample business data (sales + customer reviews)
2. Perform AI-powered sentiment analysis
3. Classify reviews by topic
4. Detect anomalies in sales data
5. Generate AI insights and summary
6. Create and save visualizations

## 📊 Output Files

- `ai_analysis_dashboard.png` - Visual dashboard with 4 key charts
- `sales_analysis.csv` - Processed sales data with anomaly flags
- `sentiment_results.csv` - Reviews with AI sentiment scores
- `classified_reviews.csv` - Reviews categorized by topic

## 🎓 Key Features for Data Analysts

### 1. **Sentiment Analysis**
```python
sentiment_analyzer = pipeline("sentiment-analysis")
result = sentiment_analyzer("Great product!")
# Output: {'label': 'POSITIVE', 'score': 0.9998}
```

### 2. **Zero-Shot Classification**
Classify text without training data:
```python
classifier = pipeline("zero-shot-classification")
result = classifier(
    "Product arrived damaged",
    candidate_labels=["Quality", "Shipping", "Service"]
)
```

### 3. **Anomaly Detection**
Statistical methods to identify unusual patterns:
```python
# Z-score based detection
z_score = (value - rolling_mean) / rolling_std
is_anomaly = abs(z_score) > 3
```

### 4. **AI-Generated Insights**
Automatically summarize findings:
```python
summarizer = pipeline("summarization")
summary = summarizer(long_text, max_length=100)
```

## 📈 Sample Results

**Sentiment Analysis:**
- Positive reviews: 16/20 (80%)
- Negative reviews: 4/20 (20%)
- Average confidence: 99.7%

**Anomaly Detection:**
- 12 anomalies detected in 730 days
- Average deviation: 3.4 standard deviations

**Topic Distribution:**
- Product Quality: 40%
- Customer Service: 25%
- Shipping: 20%
- Pricing: 15%

## 🔄Extending the Project

### Add Real Data
Replace `generate_sample_data()` with your CSV:
```python
sales_df = pd.read_csv('your_sales_data.csv')
reviews_df = pd.read_csv('your_reviews.csv')
```

### Try Different Models
```python
# For financial sentiment
analyzer = pipeline("sentiment-analysis", 
                   model="ProsusAI/finbert")

# For multilingual text
classifier = pipeline("zero-shot-classification",
                     model="facebook/mbart-large-mnli")
```

### Add Streamlit Dashboard
```python
import streamlit as st

st.title("AI Data Analysis Dashboard")
uploaded_file = st.file_uploader("Upload CSV")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    results = analyst.perform_sentiment_analysis(df)
    st.dataframe(results)
```

## 💡 Business Applications

1. **Marketing**: Analyze campaign feedback sentiment
2. **Sales**: Detect unusual sales patterns early
3. **Customer Success**: Categorize support tickets automatically
4. **Product**: Extract insights from user reviews
5. **Executive Reporting**: Generate automated summaries

