
# Python Code for Thematic Sentiment Analysis of COVID-19 Tweets

# Required Libraries
import pandas as pd 
import re 
import string
import warnings
import nltk
warnings.simplefilter(action='ignore',category=FutureWarning)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer #to stem words
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import glob
import json
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
import random
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from operator import itemgetter
import gensim
from gensim.utils import simple_preprocess
import pprint 
from sklearn.decomposition import NMF
from gensim.models.nmf import Nmf
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import emoji
nltk.download('words')
words = set(nltk.corpus.words.words())
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import snscrape.modules.twitter as sntwitter
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertTokenizer
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

os.environ['KMP_DUPLICATE_LIB_OK']='True'



# ---------------------------
# Data Preprocessing
# ---------------------------

# Loading dataset
# Combine csv files
os.chdir('/Users/Downloads/Tweets/')

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
  #combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( 'Users/Downloads/Tweets/tweets_data.csv", index=False, encoding='utf-8-sig')

mydata = pd.read_csv('/Users/Downloads/Tweets/tweets_data.csv')

# Cleaning Tweets data                    

# Keep only the 'text' column
mydata = mydata[['text']]

# Drop duplicate rows
mydata.drop_duplicates(inplace=True)

# Remove tweets that are not in English
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

mydata['is_english'] = mydata['text'].apply(is_english)
mydata = mydata[mydata['is_english']].drop(columns=['is_english'])

# Remove retweets (tweets starting with "RT")
mydata = mydata[~mydata['text'].str.startswith('RT')]

# Filter tweets related to COVID-19
covid_keywords = ['covid', 'coronavirus', 'pandemic', 'lockdown', 'quarantine', 'vaccine', 'mask', 'covid19', 'sars-cov-2']
mydata = mydata[mydata['text'].str.contains('|'.join(covid_keywords), case=False, na=False)]

# Clean text function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'@[\w_]+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove links
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words (less than 3 letters)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Reduce repeated characters (e.g., soooo -> soo)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

mydata['text'] = mydata['text'].astype(str).apply(clean_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

mydata['text'] = mydata['text'].apply(remove_stopwords)

# Reset index and save cleaned data
mydata.reset_index(drop=True, inplace=True)
df.to_csv('cleaned_tweets.csv', index=False, encoding='utf-8')

print("Tweet cleaning complete. Cleaned data saved as 'cleaned_tweets.csv'.")

    
# Listing data
mysata = [str(i) for i in mydata['text'].tolist()] # important for listing data

    
# ---------------------------
# Sentiment extraction labels with BERTweet
# ---------------------------
#Load pre-trained model and tokenizer
model = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# mydata['sentiment'] = mydata['text'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
# Tokenize and encode the tweets
tokens = tokenizer(data['text'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)

# Get model predictions
with torch.no_grad():
    outputs = model(**tokens)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_classes = predictions.argmax(dim=1)
    
# Map the 5-class predictions to 3 classes
def map_to_3_classes(prediction):
    
# Mapping 5 classes to 3: 0 and 1 as negative, 2 as neutral, 3 and 4 as positive
    if prediction in [0, 1]:
        return "negative"
    elif prediction == 2:
        return "neutral"
    else:
        return "positive"

# Apply the mapping function to the predictions
mydata['sentiment'] = [map_to_3_classes(pred.item()) for pred in predicted_classes]

# Display the results
print(mydata[['text', 'sentiment']])

#---------------------------
# Visualization of Tweets Sentiments
#---------------------------

# Count the sentiment occurrences
sentiment_counts = mydata['sentiment'].value_counts()

# Labels and sizes
labels = sentiment_counts.index
sizes = sentiment_counts.values
colors = ['#66b3ff', '#ff9999', '#99ff99']  # Blue for neutral, red for negative, green for positive
explode = (0.1, 0.1, 0.1)  # "Explode" all slices for a 3D effect

# Plotting the pseudo-3D Pie Chart
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    shadow=True,
    colors=colors,
    explode=explode
)
ax.set_title("Sentiment Distribution of Tweets (3D Pie Chart Style)", fontsize=14)
plt.tight_layout()
plt.show()


# Generate word clouds for each sentiment type

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
axes = axes.flatten()

# Combine keywords for each sentiment type
positive_keywords = " ".join(mydata[mydata['sentiment'] == 'positive']['text']).lower()
neutral_keywords = " ".join(mydata[mydata['sentiment'] == 'neutral']['text']).lower()
negative_keywords = " ".join(mydata[mydata['sentiment'] == 'negative']['text']).lower()

# Generate word clouds
sentiment_keywords = {
    'Positive Sentiment': positive_keywords,
    'Neutral Sentiment': neutral_keywords,
    'Negative Sentiment': negative_keywords
}

for i, (sentiment, keywords) in enumerate(sentiment_keywords.items()):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(keywords)
    axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].set_title(sentiment, fontsize=16)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# COVID-19 tweets sentiment analysis over time
# Attach time stamp from our tweets dataset with sentiments
# Ensure 'date' is datetime format
df['date'] = pd.to_datetime(df['date'])

# Set time granularity — monthly
mydata['month'] = mydata['date'].dt.to_period('M')

# Group by month and sentiment
sentiment_trend = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)

# Rename columns
sentiment_trend.columns = ['Negative', 'Neutral', 'Positive']

# Convert 'month' back to datetime for plotting
sentiment_trend.index = sentiment_trend.index.to_timestamp()


monthly_stamps = [
    "Jan 2020", "Feb 2020", "Mar 2020", "Apr 2020", "May 2020", "Jun 2020",
    "Jul 2020", "Aug 2020", "Sep 2020", "Oct 2020", "Nov 2020", "Dec 2020",
    "Jan 2021", "Feb 2021", "Mar 2021", "Apr 2021", "May 2021", "Jun 2021",
    "Jul 2021", "Aug 2021", "Sep 2021", "Oct 2021", "Nov 2021", "Dec 2021"
]


# Grouping by month and sentiment to calculate trends over time
sentiment_trends = stamps_data.groupby(["month", "sentiment"]).size().unstack(fill_value=0)

# Normalizing sentiment counts to calculate percentages
sentiment_trends_percentage = sentiment_trends.div(sentiment_trends.sum(axis=1), axis=0) * 100

# Sorting months chronologically for proper visualization
sentiment_trends_percentage = sentiment_trends_percentage.reindex(monthly_stamps)

# Plotting sentiment trends over time
plt.figure(figsize=(14, 8))
for sentiment in sentiment_trends_percentage.columns:
    plt.plot(sentiment_trends_percentage.index, sentiment_trends_percentage[sentiment], label=sentiment.capitalize())

plt.title("Sentiment Trends Over Time (January 2020 - December 2021)", fontsize=14)
plt.xlabel("Time (Monthly)", fontsize=12)
plt.ylabel("Sentiment Percentage (%)", fontsize=12)
plt.legend(title="Sentiment", fontsize=10, loc="upper left")
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#-----------------------------
# Fine-Tune BERTweet for Tweets Classification
#----------------------------
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load cleaned tweets dataset

# Encode sentiment labels
sentiment_encoding = {'positive': 1, 'neutral': 0, 'negative': -1}
mydata['label'] = mydata['sentiment'].map(sentiment_encoding)

# Filter for max length of 71 words
mydata['word_count'] = mydata['text'].apply(lambda x: len(x.split()))
mydata = mydata[mydata['word_count'] <= 71].reset_index(drop=True)

# Split dataset
train_texts, temp_texts, train_labels, temp_labels = train_test_split(mydata['text'], mydata['label'], test_size=0.3, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.33, random_state=42)  # 20% val, 10% test

# Load BERTweet tokenizer (uncased)
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True, normalization=True)

# Tokenize
def tokenize_data(texts, max_length):
    return tokenizer(list(texts), truncation=True, padding=True, max_length=max_length, return_tensors="pt")

# Dataset Class
class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Define experiment configurations
sequence_lengths = [32, 64]
batch_sizes = [32, 64, 128]
learning_rates = [3e-4, 1e-4, 2e-5]
weight_decay = 0.01

# Training loop
def train_model(seq_len, batch_size, lr):
    print(f"\nTraining with seq_len={seq_len}, batch_size={batch_size}, lr={lr}")
    # Tokenize
    train_encodings = tokenize_data(train_texts, seq_len)
    val_encodings = tokenize_data(val_texts, seq_len)
    test_encodings = tokenize_data(test_texts, seq_len)

    train_dataset = TweetDataset(train_encodings, train_labels)
    val_dataset = TweetDataset(val_encodings, val_labels)
    test_dataset = TweetDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=3)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = len(train_loader) * 3  # Assuming 3 epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Training loop
    model.train()
    for epoch in range(3):
        total_loss = 0
        loop = tqdm(train_loader, leave=False)
        for batch in loop:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

    # Evaluation function
    def evaluate(loader):
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds += torch.argmax(logits, axis=1).cpu().numpy().tolist()
                true += batch['labels'].cpu().numpy().tolist()
        return preds, true

    # Validation performance
    val_preds, val_true = evaluate(val_loader)
    print("Validation Performance:")
    print_metrics(val_true, val_preds)

    # Test performance
    test_preds, test_true = evaluate(test_loader)
    print("Test Performance:")
    print_metrics(test_true, test_preds)

# Evaluation metrics
def print_metrics(true, preds):
    print(f"Accuracy:  {accuracy_score(true, preds):.4f}")
    print(f"Precision: {precision_score(true, preds, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(true, preds, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(true, preds, average='weighted'):.4f}")

# Run experiments
for seq_len in sequence_lengths:
    for batch_size in batch_sizes:
        for lr in learning_rates:
            train_model(seq_len, batch_size, lr)
            

# Generate a confusion matrix using the sentiment data
# As we have actual sentiment labels and predicted labels for comparison, we'll use the 'sentiment' column for simplicity
# Actual and predicted sentiment labels (using the same data for demonstration purposes)

actual_sentiments = tweets_mydata['sentiment']
predicted_sentiments = tweets_mydata['sentiment']

# Create the confusion matrix
conf_matrix = confusion_matrix(actual_sentiments, predicted_sentiments, labels=['negative', 'neutral', 'positive'])

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Sentiment Confusion Matrix')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.tight_layout()
plt.show()

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# SVM Model
svm = Pipeline([('tfidf', vectorizer), ('svm', LinearSVC())])
svm.fit(train_texts, train_labels)
svm_preds = svm.predict(test_texts)

# Naive Bayes Model
nb = Pipeline([('tfidf', vectorizer), ('nb', MultinomialNB())])
nb.fit(train_texts, train_labels)
nb_preds = nb.predict(test_texts)

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

vocab_size = tokenizer.vocab_size
lstm_model = LSTMClassifier(vocab_size).to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# CNN Model

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=3):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(128 * 30, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

cnn_model = CNNClassifier(vocab_size).to(device)
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)

def print_report(name, y_true, y_pred):
    print(f"\n{name} Results:")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]))

print_report("SVM", test_labels + 1, svm_preds + 1)
print_report("Naive Bayes", test_labels + 1, nb_preds + 1)
print_report("LSTM", lstm_true, lstm_preds)
print_report("CNN", cnn_true, cnn_preds)
print_report("BERTweet", bert_true, bert_preds)



# ---------------------------
# Thematic Senyiment analysis with BERTopic
# ---------------------------
topic_model = BERTopic(embedding_model="paraphrase-MiniLM-L6-v2")
topics, probs = topic_model.fit_transform(df['clean_text'].tolist())
mydata['topic'] = topics

# Fine-Tune BRETopic

umap_params = {
    "n_neighbors": [3, 8, 12],
    "n_components": [2, 3, 4],
    "min_dist": [0.1, 0.01, 0.04]
}

param_combinations = list(itertools.product(
    umap_params['n_neighbors'],
    umap_params['n_components'],
    umap_params['min_dist']
))

scores = []

for n_neighbors, n_components, min_dist in tqdm(param_combinations, desc="UMAP Search"):
    umap_model = UMAP(n_neighbors=n_neighbors, 
                      n_components=n_components, 
                      min_dist=min_dist, 
                      random_state=42)

    hdb = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=30)
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdb, calculate_probabilities=False, verbose=False)
    
    topics, _ = topic_model.fit_transform(texts, embeddings)
    
    if len(set(topics)) > 1:
        reduced_embeddings = umap_model.fit_transform(embeddings)
        score = silhouette_score(reduced_embeddings, topics)
    else:
        score = -1  # Skip if only 1 cluster is found

    scores.append({
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "min_dist": min_dist,
        "silhouette": score
    })

# Convert to DataFrame
scores_mydata = pd.DataFrame(scores).sort_values(by="silhouette", ascending=False)
print(scores_df.head())

# Based on score output or your choice
best_umap = UMAP(n_neighbors=12, n_components=3, min_dist=0.04, random_state=42)
final_hdbscan = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=30)

final_topic_model = BERTopic(
    umap_model=best_umap,
    hdbscan_model=final_hdbscan,
    embedding_model=embedding_model,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = final_topic_model.fit_transform(texts)

# Split data by label
positive_text = mydata[mydata['sentiment'] == 'positive']['text'].tolist()
negative_text = mydata[mydata['sentiment'] == 'negative']['text'].tolist()
neutral_text = mydata[mydata['sentiment'] == 'neutral']['text'].tolist()

# Analyze positive tweets
positive_topics, positive_probs = topic_model.fit_transform(positive_text)
print("Positive Topics:")

# Analyze negative tweets
negative_topics, negative_probs = topic_model.fit_transform(negative_text)
print("\nNegative Topics:")

# Analyze positive tweets
neutral_topics, neutral_probs = topic_model.fit_transform(neutral_text)
print("Positive Topics:")


# Map topics back to the DataFrame
mydata.loc[mydata['sentiment'] == 'positive', 'topic'] = positive_topics
mydata.loc[mydata['sentiment'] == 'negative', 'topic'] = negative_topics
mydata.loc[mydata['sentiment'] == 'neutral', 'topic'] = neutral_topics

# Display results
print(mydata)

# Group by topics and labels to analyze themes
theme_summary = data.groupby(['sentiment', 'topic']).size().unstack(fill_value=0)

# Display the theme summary
print("Theme Summary:")

# Top 10 Themes
print("\nTop Keywords for Top 10 Themes:\n")
for topic_id in top_10_topics['Topic']:
    words = final_topic_model.get_topic(topic_id)
    keywords = ", ".join([word for word, _ in words])
    print(f"Topic {topic_id}: {keywords}")

#----------------------------
# Thematic Sentiment Visualization
#----------------------------
# Themes for word clouds
# Generate and plot a word cloud for each theme
fig, axes = plt.subplots(5, 2, figsize=(18, 20))
axes = axes.flatten()

for idx, (sentiment, topics) in enumerate(theme_keywords.items()):
    wordcloud = WordCloud(width=600, height=400, background_color="white").generate(" ".join(keywords))
    axes[idx].imshow(wordcloud, interpolation="bilinear")
    axes[idx].set_title(theme, fontsize=14)
    axes[idx].axis("off")

plt.tight_layout()
plt.show()


# Plot sentiment distribution across Themes
fig, ax = plt.subplots(figsize=(12, 8))

# Stacked bar chart
data.set_index("Theme").plot(kind="bar", stacked=True, ax=ax, width=0.8)
ax.set_title("Sentiment Distribution Across Themes", fontsize=16)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_xlabel("Themes", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.legend(title="Sentiment", fontsize=10)
plt.tight_layout()
plt.show()


# Table of Themes with associated sentiment

# The top keywords for each topic
def get_topic_keywords(topic_model, topic_id, top_n=10):
    topic = topic_model.get_topic(topic_id)
    return ", ".join([word for word, _ in topic[:top_n]])

# Associted sentiment distribution
def get_sentiment_counts(mydata, topic_id):
    subset = mydata[mydata['topic'] == topic_id]
    return {
        'Positive': (subset['sentiment'] == 1).sum(),
        'Neutral': (subset['sentiment'] == 0).sum(),
        'Negative': (subset['sentiment'] == -1).sum(),
        'Total': len(subset)
    }

# Create summary table
summary = []

# Remove outlier topic -1 if exists
valid_topics = mydata['topic'].unique()
valid_topics = [t for t in valid_topics if t != -1]

for topic_id in valid_topics:
    keywords = get_topic_keywords(final_topic_model, topic_id)
    sentiments = get_sentiment_counts(df, topic_id)

    summary.append({
        'Topic ID': topic_id,
        'Top Keywords': keywords,
        'Total Tweets': sentiments['Total'],
        'Positive': sentiments['Positive'],
        'Neutral': sentiments['Neutral'],
        'Negative': sentiments['Negative']
    })

summary_mydata = pd.DataFrame(summary)
summary_mydata = summary_mydata.sort_values(by='Total Tweets', ascending=False).reset_index(drop=True)

# Display the summary table
print(summary_mydata.head(10))

# Bar Chart for sentiment distribution across themes

Convert data to a DataFrame for better readability
sentiment_mydata = pd.DataFrame(sentiment_data, columns=["Positive", "Negative", "Neutral"], index=themes)

# Transpose data for plotting
sentiment_transposed = sentiment_df.T

# Plot bar chart for sentiment distribution across themes
sentiment_transposed.plot(kind="bar", figsize=(14, 8), width=0.8)
plt.title("Sentiment Distribution Across Themes", fontsize=16)
plt.xlabel("Sentiments", fontsize=14)
plt.ylabel("Percentage (%)", fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.legend(title="Themes", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
plt.tight_layout()
plt.show()


# Themes Across Different Pandemic Phases visualization
phases = [
    "Early Pandemic (Jan 2020 - May 2020)",
    "Middle Pandemic (Jun 2020 - Dec 2020)",
    "Vaccine Rollout (Jan 2021 - May 2021)",
    "Recovery & Variants (Jun 2021 - Dec 2021)"
]
phase_months = [
    months[:5],  # Jan 2020 - May 2020
    months[5:12],  # Jun 2020 - Dec 2020
    months[12:17],  # Jan 2021 - May 2021
    months[17:]  # Jun 2021 - Dec 2021
]

# Assign phases to the events
events_mydata["Phase"] = ""
for phase, months_range in zip(phases, phase_months):
    events_df.loc[events_mydata["Month"].isin(months_range), "Phase"] = phase

# Count themes per phase
theme_phase_distribution = events_df.groupby(["Phase", "Theme"]).size().unstack(fill_value=0)

# Plot the distribution of themes across phases
theme_phase_distribution.plot(kind="bar", stacked=True, figsize=(14, 8))

# Customize the plot
plt.title("Themes Across Different Pandemic Phases", fontsize=16)
plt.xlabel("Pandemic Phases", fontsize=12)
plt.ylabel("Number of Events", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.legend(title="Themes", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
plt.tight_layout()
plt.show()

# Network Graph
G = nx.Graph()
for theme, words in keywords.items():
    G.add_node(theme, type='theme')
    for word in words:
        G.add_node(word, type='keyword')
        G.add_edge(theme, word)
for theme in themes:
    for sentiment in sentiments:
        G.add_edge(theme, sentiment)

# Plot Network Graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue')
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
plt.title("Network Graph of Themes, Keywords, and Sentiments", fontsize=16)
plt.show()

# Heatmap
# Set number of top themes to show in the heatmap
TOP_N = 10

# Prepare data for heatmap
heatmap_data = summary_mydata.head(TOP_N)[['Topic ID', 'Positive', 'Neutral', 'Negative']]

# Set 'Topic ID' as index
heatmap_data.set_index('Topic ID', inplace=True)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5, linecolor='gray')

plt.title("Sentiment Distribution Across Top Topics", fontsize=14)
plt.ylabel("Topic ID")
plt.xlabel("Sentiment")
plt.tight_layout()
plt.show()

# The similarity heatmap for themes Plot
plt.figure(figsize=(12, 10))
sns.heatmap(theme_correlation_matrix, annot=True, cmap="coolwarm", cbar_kws={'label': 'Similarity Coefficient'}, fmt=".2f")
plt.title("Themes Correlation Heatmap Based on Sentiment Similarity", fontsize=16)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=0)
plt.tight_layout()
plt.show()




