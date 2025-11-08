# Thematic Sentiment Analysis Of COVID-19 tweets
This repository contains the implementation and workflow of a Thematic Sentiment Analysis (TSA) framework. 
The project combines thematic sentiment using state-of-the-art NLP models namely BERTopic and BERTweet to 
extract meaningful public insights from large-scale Twitter data collected during the COVID-19 pandemic.

# Overview
	This analysis explore public sentiment and themes expressed on social media during the COVID-19 pandemic. 
	We used about 327 million tweets collected between January 2020 – December 2021, using 15 COVID-19-related keywords. 
	Empolyed Fine-tuned BERTweet model for sentiment classification (positive, negative, neutral) and BERTopic for 
	thematic (topic) analysis and clustering.


# Methodology
  1. Data Preprocessing
	•	Cleaning, language filtering, tokenisation, and stopword removal.
	2.	Sentiment Classification
	•	Fine-tuned BERTweet-base using labelled COVID-19 tweet subsets.
	•	Comparison with traditional ML models (SVM, Naïve Bayes) and DL models (LSTM, CNN).
	3.	Thematic Analysis
	•	BERTopic to extract and visualise emerging topics.
	•	Dimensionality reduction via UMAP and clustering using HDBSCAN.
	4.	Integration
	•	Merged sentiment labels with thematic clusters for Thematic Sentiment Analysis (TSA).
	
# Key Features
	•	End-to-end pipeline for large-scale social media analysis.
	•	Hybrid integration of transformer-based sentiment analysis and topic modeling.
	•	Support for visual analytics (word clouds, sentiment distribution plots, topic evolution).
	•	Reproducible and scalable with modular Python scripts.



# Technologies Used
	•	Python, PyTorch, Transformers (Hugging Face)
	•	BERTweet, BERTopic, UMAP, HDBSCAN
	•	Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn



# Outputs
	•	Sentiment-labelled datasets
	•	Topic–sentiment correlation visualisations
	•	Thematic insights for policymaking and public communication


## 📊 Sentiment Distribution
Sentiment Distribution

## 🗂️ Top Themes
| Theme | Keywords | Positive | Neutral | Negative |
|--------|-----------|-----------|-----------|-----------|
| Health Policy | vaccine, nhs, lockdown | 42% | 25% | 33% |


# Citation
If you use this repository, please cite:
Yahia, A. (2025). Mine Social Media to Gain Insight into Pandemic: Thematic Sentiment Analysis of COVID-19 Tweets.
