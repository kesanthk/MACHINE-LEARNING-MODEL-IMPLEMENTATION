# MACHINE-LEARNING-MODEL-IMPLEMENTATION

# Step 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Step 2: Load Dataset (Use built-in or CSV)
# Using the SMS Spam Collection Dataset from UCI
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', names=["label", "message"])
df.head()

# Step 3: Preprocessing
df['label_num'] = df.label.map({'ham':0, 'spam':1})  # Convert labels to binary
X = df['message']
y = df['label_num']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Vectorization (Convert text to numbers)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test_vec)

# Step 8: Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))