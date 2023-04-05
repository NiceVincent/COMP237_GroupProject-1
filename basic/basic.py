#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:46:54 2023

@author: Vincent
"""

import pandas as pd
import os
path = "./"
filename = 'Youtube02-KatyPerry.csv'
fullpath = os.path.join(path,filename)
df = pd.read_csv(fullpath,sep=',',encoding='ISO-8859-1')

# basic exploration of the data
#print(f'head : \n {df.head(2)}')
#print('--------')
print(f'shape : \n {df.shape}')
print('--------')
print(f'columns : \n {df.columns}')
print('--------')
print(f'info : \n {df.info()}')
print(f'null : \n {df.isnull().sum()}')
#print(f'describe : \n {df.describe()}')
#print(df.tail(3))
print('--------')
print(f"CONTENT col data : \n {df['CONTENT']}")
print('--------')

# prepare the data using nltk and count_vectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

# vectorize get words freq array
count_vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
X = count_vectorizer.fit_transform(df['CONTENT'])
#print(f'X data is => \n {X}')
Y = df['CLASS']
#print(f'Y data is => \n {Y}')

# present highlights of initial features
print(f'feature data shape : \n {X.shape}')
# print(f'X data shape 0 : \n {X}')

# downscale the data using tf-idf
#  transform and weight the token value
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X)
print("Transformed Data Shape:", X_tfidf.shape)
# print(f'X_tfidf data shape 1 : \n {X_tfidf}')

# test, train data split and shuffle
# requirement 6, 7 - method 1
# shuffle the dataset
#data_shuffled = df.sample(frac=1, random_state=1)
#X_shuffled = X_tfidf[data_shuffled.index, :]
#y_shuffled = Y[data_shuffled.index]

# split the dataset into training and testing sets
# train_size = int(0.75 * len(data_shuffled))

# X_train = X_shuffled[:train_size, :]
# y_train = y_shuffled[:train_size]
# X_test = X_shuffled[train_size:, :]
# y_test = y_shuffled[train_size:]

# requirement 6, 7 - method 2
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X_tfidf, Y, test_size=0.25, random_state=1)
    

# fit the Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train, y_train)

# cross validate the model
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Mean accuracy:", cv_scores.mean())


# test the model and show the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred))

# plot the confusion matrix
cm_df = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm_df, annot=True, fmt='d')
plt.xlabel('Predict Values')
plt.ylabel('Actual Values')

# test the classifier with new comments
new_comments = [
    "Great movie, I enjoyed it!",
    "The storyline was fantastic!",
    "This movie was a waste of time.",
    "I wouldn't recommend this movie.",
    "Watch now for free at freemovies123!",
    "Click here to win a free iPhone!",
]

new_comments_transformed = count_vectorizer.transform(new_comments)
new_comments_tfidf = tfidf.transform(new_comments_transformed)
predictions = clf.predict(new_comments_tfidf)

print("Predicted classes:", predictions)