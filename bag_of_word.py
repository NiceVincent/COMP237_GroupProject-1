
import os
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


# def word_tokenize_and_remove_stop_word(text):
#     token = nltk.word_tokenize(text)
#     return [x.lower() for x in token if (x.isalpha() and x not in stopwords.words('english'))]


def word_Lemmatizer(texts):
    return ''.join([nltk.WordNetLemmatizer().lemmatize(word=x)for x in texts])


def word_PorterStemmer(texts):
    return ''.join([nltk.PorterStemmer().stem(word=x)for x in texts])


def word_SnowballStemmer(texts):
    return ''.join([nltk.SnowballStemmer(language='english').stem(x)for x in texts])


def word_LancasterStemmer(texts):
    return ''.join([nltk.LancasterStemmer().stem(word=x)for x in texts])


def count_vectorizer(data):
    # Change mid_df for adjust nmber of vocabulary will be use, if the word frequency lower than min_df, it won't showw
    count_vectorizer = CountVectorizer(stop_words='english', min_df=6)
    bag_of_words = count_vectorizer.fit_transform(
        [content for content in data['CONTENT']])
    return pd.DataFrame(bag_of_words.toarray(
    ), columns=count_vectorizer.get_feature_names_out())


# Load YouTube spam data in dataframe
directory = os.path.dirname(os.path.realpath(__file__))
directory = os.path.join(directory, "YouTube-Spam-Collection-v1")
files = [f for f in os.listdir(directory)]

# Create dataframe
# Columns: COMMENT_ID, AUTHOR, DATE, CONTENT, TAG
yt_data = pd.concat((pd.read_csv(os.path.join(directory, f)) for f in files))

# Show is any missing data
print(yt_data.isna().any())
print(yt_data.isnull().any())

# There are total 1005 message classified as span
print('Number of spam message: ', yt_data['CLASS'].where(
    yt_data['CLASS'] > 0).count())
print('Number of ham message: ', yt_data['CLASS'].where(
    yt_data['CLASS'] == 0).count())

print('Longest message: ', yt_data['CONTENT'].str.len().max())

# Content and class matter, keep two columns only
train_data = yt_data[['CONTENT', 'CLASS']]
print('Train data')
print(train_data)

# Pre-process
# We try different approach inorder to find best approach for our model

# train_data['CONTENT'] = train_data.apply(
#     lambda x: word_tokenize_and_remove_stop_word(x['CONTENT']), axis=1)
# print(train_data.head())

train_data_lemmatize = train_data.copy()
train_data_SnowballStemming = train_data.copy()
train_data_LancasterStemming = train_data.copy()
train_data_PorterStemming = train_data.copy()

train_data_lemmatize['CONTENT'] = train_data.apply(
    lambda x: word_Lemmatizer(x['CONTENT']), axis=1)
print(train_data_lemmatize.head())

train_data_LancasterStemming['CONTENT'] = train_data.apply(
    lambda x: word_LancasterStemmer(x['CONTENT']), axis=1)
print(train_data_LancasterStemming.head())

train_data_SnowballStemming['CONTENT'] = train_data.apply(
    lambda x: word_SnowballStemmer(x['CONTENT']), axis=1)
print(train_data_SnowballStemming.head())

train_data_PorterStemming['CONTENT'] = train_data.apply(
    lambda x: word_PorterStemmer(x['CONTENT']), axis=1)
print(train_data_PorterStemming.head())

# Extract the document term matrix
train_data_lemmatize_bog = count_vectorizer(train_data_lemmatize).head()
train_data_LancasterStemming_bog = count_vectorizer(
    train_data_LancasterStemming).head()
train_data_SnowballStemming_bog = count_vectorizer(
    train_data_SnowballStemming).head()
train_data_PorterStemming_bog = count_vectorizer(
    train_data_PorterStemming).head()
