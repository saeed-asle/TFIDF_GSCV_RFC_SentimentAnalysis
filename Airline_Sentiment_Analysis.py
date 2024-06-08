import numpy as np
import pandas as pd
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Load the dataset
airline_tweets = pd.read_csv('Tweets.csv')

# Data Exploration and Visualization
def visualize_data():
    # Set plot size for better visualization
    plt.figure(figsize=(12, 6))

    # Pie chart showing airline distribution
    plt.subplot(121)
    airline_tweets['airline'].value_counts().plot(kind='pie', autopct='%1.0f%%')
    plt.title('Airline Distribution')

    # Pie chart showing sentiment distribution
    plt.subplot(122)
    airline_tweets['airline_sentiment'].value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])
    plt.title('Sentiment Distribution')

    # Bar chart showing sentiment distribution for each airline
    plt.figure(figsize=(12, 6))
    sns.countplot(x='airline', hue='airline_sentiment', data=airline_tweets)
    plt.title('Sentiment Distribution by Airline')

# Text Preprocessing
def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    # Substituting multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Data Preprocessing
airline_tweets['text'] = airline_tweets['text'].apply(preprocess_text)

# Feature Extraction (TF-IDF)
def tfidf_vectorization(text_data):
    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
    return vectorizer.fit_transform(text_data).toarray()

# Splitting Data
X = tfidf_vectorization(airline_tweets['text'])
y = airline_tweets['airline_sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model Training (Random Forest Classifier)
def train_random_forest_classifier(X_train, y_train):
    classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    classifier.fit(X_train, y_train)
    return classifier

# Hyperparameter Tuning with Grid Search
def tune_classifier(X_train, y_train):
    param_grid = {
        'n_estimators':[100, 200,300],
        'max_depth':[10, 20,30, None],
        'min_samples_split':[2, 5,10],
        'min_samples_leaf':[1,2, 4]
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

visualize_data()  # Visualize data
plt.show()  # Show plots

# Train a Random Forest classifier
classifier = train_random_forest_classifier(X_train, y_train)

# Hyperparameter Tuning with Grid Search
# classifier = tune_classifier(X_train, y_train)

# Evaluate the model
predictions = classifier.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
