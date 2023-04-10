#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 6 11:26:14 2023

@author: chinnawut-b
"""
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score


def load_data(dir_name):
    data = []
    for file in os.listdir(dir_name):
        fullpath = os.path.join(dir_name, file)
        result = pd.read_csv(fullpath, sep=',', encoding='ISO-8859-1')
        data.append(result)
    return pd.concat(data)

def preprocess_text(data, count_vectorizer=None):
    if count_vectorizer is None:
        count_vectorizer = CountVectorizer(stop_words=stopwords.words('english'), min_df=1)
        x_content = count_vectorizer.fit_transform(data.to_numpy())
    else:
        x_content = count_vectorizer.transform(data.to_numpy())
    return x_content, count_vectorizer

# process text with lemmatization and tokenization
# def preprocess_text(data, count_vectorizer=None):
    # # Initialize the WordNetLemmatizer
    # lemmatizer = WordNetLemmatizer()
    
    # # Define a function to tokenize and lemmatize text
    # def tokenize_and_lemmatize(text):
    #     tokens = word_tokenize(text)
    #     return [lemmatizer.lemmatize(token) for token in tokens]
    
    # if count_vectorizer is None:
    #     count_vectorizer = CountVectorizer(stop_words=stopwords.words('english'), min_df=1, tokenizer=tokenize_and_lemmatize)
    #     x_content = count_vectorizer.fit_transform(data.to_numpy())
    # else:
    #     x_content = count_vectorizer.transform(data.to_numpy())
    # return x_content, count_vectorizer

# Train data group file Youtube02-KatyPerry.csv
train_data = load_data("/Users/chinnawut-b/centennialcollege/AI/GROUP_PROJECT/COMP237_GroupProject/TrainData")

# Test data all file in TestData Directory
test_data = load_data("/Users/chinnawut-b/centennialcollege/AI/GROUP_PROJECT/COMP237_GroupProject/TestData")

print('shape: \n', train_data.shape)
print('--------')
print('columns: \n', train_data.columns)
print('--------')
print('info: \n', train_data.info())
print('--------')
print('null value in column: \n', train_data.isnull().sum())
print('--------')
print("column CONTENT data: \n ", train_data['CONTENT'])
print('--------')


X_train_data, count_vectorizer = preprocess_text(train_data['CONTENT'])

Y_train_data = train_data['CLASS']

print('shape X_train CountVectorizer: \n', X_train_data.shape)
print('--------')


# using Term Frequency transform
tfidf = TfidfTransformer()
X_tfidf_train_data = tfidf.fit_transform(X_train_data)
print("Transformed Data Shape:", X_tfidf_train_data.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf_train_data, Y_train_data, test_size=0.25, random_state=64)

train_data_classifier = MultinomialNB().fit(X_train, Y_train)

# Cross validate the model on the training data using 5-fold
cv_scores = cross_val_score(train_data_classifier, X_train, Y_train, cv=5)
print("Mean accuracy:", cv_scores.mean())

# Test the model on the test data
Y_pred = train_data_classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy_score(Y_test, Y_pred))


# Preprocess the new data frame
X_test_data, count_vectorizer_test_data = preprocess_text(test_data['CONTENT'], count_vectorizer)
X_test_data_tfidf = tfidf.transform(X_test_data)
Y_test_data = test_data['CLASS']

print("Transformed Test Data Shape:", X_test_data_tfidf.shape)

Y_test_data_pred = train_data_classifier.predict(X_test_data_tfidf)
cm_new = confusion_matrix(Y_test_data, Y_test_data_pred)

print("Confusion Matrix:\n", cm_new)
print("Accuracy:", accuracy_score(Y_test_data, Y_test_data_pred))

