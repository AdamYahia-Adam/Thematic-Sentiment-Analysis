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


<img width="1024" height="1536" alt="A_flowchart_in_the_digital_vector_graphic_illustra" src="https://github.com/user-attachments/assets/9a49b49f-2970-45b4-b5b2-65cf03e8eeef" />

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

Pie chart reveals that about 40% of the tweets expressed positive sentiments, where Negative sentiments about 36%, which is 
slightly lower than the positive sentiment and the neutral about 24%.

<img width="296" height="251" alt="image" src="https://github.com/user-attachments/assets/51dc6659-906d-4fc4-8c7e-b5d2f420d162" />


The table of classification report bellow shows that, the model is very good at identifying positive, negative, and neutral sentiments.
The experiment results of fine-tune BERTweet model classified COVID-19 tweet datasets with accuracy of 95.6% for training dataset, 94.3% 
for validating dataset and 91.8% for testing dataset. 

Sentiment Classification Performance
| Class     | Precision | Recall | F1-Score | Support   |
|------------|------------|--------|----------|-----------|
| Positive   | 0.97       | 0.95   | 0.96     | 5,262,236 |
| Negative   | 0.96       | 0.96   | 0.96     | 4,544,489 |
| Neutral    | 0.94       | 0.96   | 0.94     | 2,391,250 |



Figure bellow presents a word cloud that represents the most frequently used words in tweets expressing negative, positive, and neutral sentiments, 
along with a combined word cloud of all sentiments.

<img width="3980" height="737" alt="output" src="https://github.com/user-attachments/assets/26677743-0ec2-4b58-8eff-426c2a08f319" />

The line chart presented below, illustrates sentiment trends over time from January 2020 to December 2021. In negative Sentiment we observed some 
variations, often peaking during months associated with major pandemic events or lockdown announcements but neutral sentiment remained relatively 
stable throughout the period where positive sentiment shows change, with some peaks indicating moments of hopeful news, possibly linked to vaccine 
rollouts or easing restrictions.



<img width="2779" height="1579" alt="Sentiment Trends Over Time (Jan 2020 - Dec 2021)" src="https://github.com/user-attachments/assets/92b07db4-4fe9-4575-a0d6-3dd0e8ada50e" />

Extracted sentiment with its scores and themes, then we aggregated sentiment scores by theme and summarize the overall sentiment within each identified
Theme as shown in Table bellow. These scores associated with themes provide valuable insights from each theme. For example: the theme "Vaccination," the
aggregate sentiment score might reflect the proportion of positive, neutral, and negative tweets related to vaccinations.  

| Theme                     | Positive | Negative | Neutral |
|----------------------------|-----------|-----------|-----------|
| Public Health Measures     | 20%       | 55%       | 25%       |
| Vaccine Development        | 45%       | 30%       | 25%       |
| Economic Impact            | 10%       | 70%       | 20%       |
| Mental Health              | 15%       | 60%       | 25%       |
| Misinformation             | 5%        | 80%       | 15%       |
| Gratitude and Support      | 85%       | 5%        | 10%       |
| Global Statistics          | 25%       | 50%       | 25%       |
| Education and Remote Work  | 30%       | 45%       | 25%       |
| Travel Restrictions        | 15%       | 70%       | 15%       |
| Political Response         | 20%       | 65%       | 15%       |



Network graph Figure bellow visualize connections between emerging themes, sentiments, and public discourse through keywords. 
Nodes represent themes, keywords, and sentiments. Where the edges represent co-occurrence or association strength e.g., a keyword 
related to multiple themes.


<img width="3600" height="3600" alt="Network_Graph_Themes_Keywords_Sentiments" src="https://github.com/user-attachments/assets/ccb41be2-cd45-49a7-8383-9ff2b55d5890" />



Result from the Figure bellow above highlighted four main phases during the COVID-19 pandemic. 

<img width="2345" height="1580" alt="output-2" src="https://github.com/user-attachments/assets/c86a61b3-16ee-4087-ba8c-2b75481938a8" />



The following Figure visualize the word clouds for all ten themes. Each theme is displayed in its own section with its associated keywords.



<img width="2665" height="3980" alt="output-8" src="https://github.com/user-attachments/assets/a35838dc-5817-4467-8121-57d7a0e5097a" />


 The Themes Correlation Heatmap bellow representing how similar or dissimilar themes are based on their sentiment distributions 
 as positive, negative, and neutral sentiments. Each cell represents the correlation coefficient between two themes. Values range
 from (-1) represent perfectly inversely correlated to (1) as perfectly correlated. Darker or more intense colours indicate stronger correlations.


<img width="307" height="323" alt="Picture 1" src="https://github.com/user-attachments/assets/43a0fe43-2864-4134-9f39-de63ce514bd6" />


 # Discussion 
 
 The analysis conducted provides a comprehensive understanding of public sentiment and thematic responses during the COVID-19 pandemic. 
 By examining themes with their sentiment distributions, correlations, and responses to major events, we were identified various key 
 patterns and their implications as shown in following points:
 
•	Positive sentiment was driven by significant events such as vaccine approvals, rollouts, and booster shot campaigns. Themes like Gratitude 
and Support shows high positivity, reflecting public appreciation for healthcare workers and scientific advancements. On the other hand, negative 
sentiment dominated themes like Misinformation and Economic Impact, which were associated with widespread public frustration and challenges such 
as job losses, financial instability, and the amplification of conspiracy theories.

•	Thematic responses during specific periods revealed how events shaped public discussion on social media. For example, during the early lockdowns
in March 2020, themes such as Public Health Measures and Gratitude and Support dominated discussions, with moderate positivity reflecting compliance 
and solidarity. However, as lockdowns extended, negative sentiment increased in themes like Mental Health and Economic Impact, highlighting the emotional
stress and financial challenges faced by individuals and communities.

•	The emerge of the Delta variant in mid 2021 brought significant negativity across themes such as Travel Restrictions and Misinformation, driven by
public fear and unhappiness about new rules. In contrast, the Omicron variant in late 2021 produced more balanced sentiment, with higher positivity linked
to widespread vaccine availability and booster campaigns. This contrast underscores the importance of preparation and public trust in reducing negativity 
during health crises.

•	Correlation analysis showed strong associations between themes like Public Health Measures, Economic Impact, and Mental Health, indicating shared drivers 
of public sentiment, such as restrictions and financial stressors. In contrast, Gratitude and Support stood out as a distinct theme, primarily associated with
positivity, highlighting its unique role in building strength and community morale.

•	Misinformation emerged as a repeated challenge, with high negative sentiment reflecting public frustration with false narratives about vaccines, public 
health measures, and the pandemic’s origins. Its influence was particularly evident during vaccination phases and variant surges, where it fuelled vaccine 
hesitancy and resistance to health measures.

•	Public adaptation over time was evident in the sentiment patterns. Initial compliance and optimism during early lockdowns gave way to weakness and frustration
as the pandemic continued. Vaccination achievements acted as turning points, shifting sentiment positively despite persistent challenges like misinformation.

In general, the results highlight the dynamic relationship between public sentiment, key events, and thematic discourse. That underscore the need for effective 
communication, trust building, and targeted interventions to address misinformation, support mental health, and sustain public morale during crises. These insights
provide a foundation for understanding societal responses to health emergencies and improving strategies for future crises.

 
# Citation
If you use this repository, please cite:
A. Yahia. (2025). Mine Social Media to Gain Insight into Pandemic: Thematic Sentiment Analysis of COVID-19 Tweets.
