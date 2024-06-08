# Airline Sentiment Analysis
Authored by saeed asle
# Description
This project performs sentiment analysis on tweets about airlines using a Random Forest classifier.
The dataset contains tweets labeled with sentiments such as positive, negative, and neutral.

The project includes the following steps:
* Data Exploration and Visualization: Visualizes the distribution of airlines and sentiments in the dataset.
* Text Preprocessing: Cleans the text data by removing special characters, single characters, and converting to lowercase.
* Feature Extraction: Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to extract features from the text data.
* Model Training: Trains a Random Forest classifier on the preprocessed text data.
* Hyperparameter Tuning: Optionally performs hyperparameter tuning using Grid Search to find the best parameters for the classifier.
* Evaluation: Evaluates the model's performance using confusion matrix, classification report, and accuracy score.
# Features
  * Data visualization: Pie charts and bar charts to visualize the distribution of airlines and sentiments.
  * Text preprocessing: Removes special characters, single characters, and converts text to lowercase.
  * Feature extraction: Uses TF-IDF vectorization to extract features from text data.
  * Model training: Trains a Random Forest classifier to predict sentiment.
  * Hyperparameter tuning: Optionally tunes the Random Forest classifier using Grid Search to improve performance.
# Dependencies
  * numpy: For numerical operations.
  * pandas: For data manipulation and analysis.
  * re: For regular expressions.
  * nltk: For natural language processing tasks.
  * seaborn: For data visualization.
  * matplotlib: For plotting graphs.
  * sklearn: For machine learning utilities.
# How to Use
* Ensure you have the necessary libraries installed, such as numpy, pandas, nltk, seaborn, matplotlib, and scikit-learn.
* Download the dataset containing tweets about airlines. You can find the dataset here.
* Run the provided code to preprocess the data, train the Random Forest classifier, and evaluate its performance.

# Output
The code outputs visualizations of the data distribution, a confusion matrix, classification report, and accuracy score for the trained model.


