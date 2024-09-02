# Twitter-Sentimental-Analysis-Using-NLP
Objective: The goal of Twitter sentiment analysis using Natural Language Processing (NLP) is to analyze the emotions, opinions, or attitudes expressed in tweets. This is commonly used to understand public sentiment towards a specific topic, product, event, or individual by categorizing tweets as positive, negative, or neutral.

Key Steps in the Process:
Data Collection:

Kaggle: Using the Kaggle dataset API to collect tweets related to specific keywords, hashtags, or accounts. 

Data Preprocessing: Clean the collected tweets by removing irrelevant data such as URLs, special characters, emojis, stopwords, and handling case normalization. Tokenization is performed to split the text into individual words or tokens.


Text Preprocessing:
Tokenization: Splitting the tweet text into individual words or tokens for further processing.

Stopword Removal: Removing common words (like "and", "the", "is") that don't contribute much to the sentiment analysis.

Stemming/Lemmatization: Reducing words to their root form (e.g., "running" to "run") to ensure consistency in word representation.

Vectorization: Converting text into numerical features that machine learning models can process. This is typically done using techniques like Bag of Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), or word embeddings (e.g., Word2Vec, GloVe).


Sentiment Analysis:
Model Selection: Logistic Regression is used for text classification.

Training: Training the model on labeled datasets where the sentiment (positive, negative, neutral) of tweets is known.
In this case, we only have positive and negative tweets equally distributed. Positive tweets are labelled as 4 which we then convert to 1 and negative tweets are labelled as 0.

Prediction: Applying the trained model to predict the sentiment of new, unseen tweets.


Evaluation:
Metrics: We evaluate the model's performance using accuracy, precision, and confusion matrices to ensure the sentiment analysis is reliable.

Validation: For validation, we perform cross-validation to ensure the model generalizes well to new data.
