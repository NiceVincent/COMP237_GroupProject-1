
from enum import Enum
import os
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


class PreprocessName(Enum):
    LEMMATIZE = "lemm"
    PORTER_STEMMER = "porter"
    SNOWBALL_STEMMER = "snowball"
    LANCASTER_STEMMER = "lancaster"


class Preprocess:

    def __init__(self):
        self.data = self.load_data()

    def load_data(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        directory = os.path.join(directory, "YouTube-Spam-Collection-v1")
        files = [f for f in os.listdir(directory)]

        # Create dataframe
        # Columns: COMMENT_ID, AUTHOR, DATE, CONTENT, TAG
        result = pd.concat(
            (pd.read_csv(os.path.join(directory, f)) for f in files))

        # Content and class matter, keep two columns only
        result = result[['CONTENT', 'CLASS']]

        # Show is any missing data
        print("============ Load File ============")
        print("Has na: data", result.isna().any())
        print("Has null data: ", result.isnull().any())
        print("Preview")
        print(result.head())
        return result

    def word_Lemmatizer(self, texts):
        return ''.join([nltk.WordNetLemmatizer().lemmatize(word=x)for x in texts])

    def word_PorterStemmer(self, texts):
        return ''.join([nltk.PorterStemmer().stem(word=x)for x in texts])

    def word_SnowballStemmer(self, texts):
        return ''.join([nltk.SnowballStemmer(language='english').stem(x)for x in texts])

    def word_LancasterStemmer(self, texts):
        return ''.join([nltk.LancasterStemmer().stem(word=x)for x in texts])

    def dataframe_Lemmatizer(self):
        tmp = self.data.copy()
        tmp['CONTENT'] = tmp.apply(
            lambda x: self.word_Lemmatizer(x['CONTENT']), axis=1)
        return tmp

    def dataframe_PorterStemmer(self):
        tmp = self.data.copy()
        tmp['CONTENT'] = tmp.apply(
            lambda x: self.word_PorterStemmer(x['CONTENT']), axis=1)
        return tmp

    def dataframe_SnowballStemmer(self):
        tmp = self.data.copy()
        tmp['CONTENT'] = tmp.apply(
            lambda x: self.word_SnowballStemmer(x['CONTENT']), axis=1)
        return tmp

    def dataframe_LancasterStemmer(self):
        tmp = self.data.copy()
        tmp['CONTENT'] = tmp.apply(
            lambda x: self.word_LancasterStemmer(x['CONTENT']), axis=1)
        return tmp

    def count_vectorizer(self, data):
        # Change mid_df for adjust nmber of vocabulary will be use, if the word frequency lower than min_df, it won't showw
        count_vectorizer = CountVectorizer(stop_words='english', min_df=6)
        bag_of_words = count_vectorizer.fit_transform(
            [content for content in data['CONTENT']])
        return pd.DataFrame(bag_of_words.toarray(
        ), columns=count_vectorizer.get_feature_names_out())

    def padding():
        return ''

    def get_tokenlized_df(self, preprocess: PreprocessName):
        df = None
        if preprocess == PreprocessName.LANCASTER_STEMMER:
            df = self.dataframe_LancasterStemmer()
        elif preprocess == PreprocessName.LEMMATIZE:
            df = self.dataframe_Lemmatizer()
        elif preprocess == PreprocessName.PORTER_STEMMER:
            df = self.dataframe_PorterStemmer()
        elif preprocess == PreprocessName.SNOWBALL_STEMMER:
            df = self.dataframe_SnowballStemmer()
        return self.count_vectorizer(df)
