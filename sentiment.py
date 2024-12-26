import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.special import softmax

# Load Models
@st.cache_resource
def load_roberta():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

def analyze_sentiment_roberta(text):
    tokenizer, model = load_roberta()
    encoded_text = tokenizer(text,  return_tensors='pt', padding=True, truncation=True,  max_length=512,
        add_special_tokens=True ) 
    output = model(**encoded_text)
    scores = softmax(output[0][0].detach().numpy())
    return {
        "Negative": scores[0],
        "Neutral": scores[1],
        "Positive": scores[2]
    }

def analyze_sentiment_vader(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return {
        "Negative": scores['neg'],
        "Neutral": scores['neu'],
        "Positive": scores['pos']
    }

# App Title and Description
st.title("Sentiment Analysis Dashboard")
st.markdown("""
Welcome to the **Sentiment Analysis Dashboard**! This app uses state-of-the-art models to analyze text sentiment. 

### How it works:
1. Input text or upload a dataset for analysis.
2. Choose between two sentiment analysis models:
   - **VADER** (a rule-based approach).
   - **RoBERTa** (a deep learning-based approach).
3. Visualize and compare results.
""")

# Input Options
st.subheader("Analyze Single Text Input")
user_input = st.text_area("Enter text here:")
if st.button("Analyze Text"):
    if user_input.strip():
        st.write("### Results")
        st.write("**VADER Analysis**:")
        vader_result = analyze_sentiment_vader(user_input)
        st.json(vader_result)

        st.write("**RoBERTa Analysis**:")
        roberta_result = analyze_sentiment_roberta(user_input)
        st.json(roberta_result)

        # Bar Chart Comparison
        st.write("### Comparison Chart")
        fig, ax = plt.subplots()
        labels = ["Negative", "Neutral", "Positive"]
        vader_scores = [vader_result[l] for l in labels]
        roberta_scores = [roberta_result[l] for l in labels]
        x = range(len(labels))

        ax.bar(x, vader_scores, width=0.4, label="VADER", align='center')
        ax.bar([p + 0.4 for p in x], roberta_scores, width=0.4, label="RoBERTa", align='center')
        ax.set_xticks([p + 0.2 for p in x])
        ax.set_xticklabels(labels)
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("Please enter some text.")

# Batch Analysis
st.subheader("Batch Analysis from File")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Show the first few rows of the dataset for the user to view and check columns
    st.write("### Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Let the user select the column that contains the text data
    text_column = st.selectbox("Select the column that contains text data", df.columns)
    
    if st.button("Analyze Dataset"):
        if text_column:
            st.write("### Analyzing Dataset...")
            results = []
            for text in df[text_column]:
                vader_result = analyze_sentiment_vader(text)
                roberta_result = analyze_sentiment_roberta(text)
                combined_result = {
                    "Text": text,
                    **vader_result,
                    **roberta_result
                }
                results.append(combined_result)

            results_df = pd.DataFrame(results)
            st.write("### Results")
            st.dataframe(results_df)

            # Visualization
            st.write("### Overall Sentiment Distribution")
            sentiment_means = results_df[["Negative", "Neutral", "Positive"]].mean()
            fig, ax = plt.subplots()
            sentiment_means.plot(kind='bar', ax=ax)
            ax.set_title("Average Sentiment Scores")
            st.pyplot(fig)

            # Download Option
            st.download_button(
                label="Download Results as CSV",
                data=results_df.to_csv(index=False),
                file_name="sentiment_results.csv",
                mime="text/csv"
            )
        else:
            st.error("Please select a column with text data.")
