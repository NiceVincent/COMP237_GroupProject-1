
import os
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import torch
from Model import Model

from Preprocess import Preprocess, PreprocessName

RANDOMS_SEED = 420
TEST_SIZE = 0.4

# Preprocess
preprocess = Preprocess()
lancaster_df = preprocess.get_tokenlized_df(PreprocessName.LANCASTER_STEMMER)
lemmatize_df = preprocess.get_tokenlized_df(PreprocessName.LEMMATIZE)
porter_df = preprocess.get_tokenlized_df(PreprocessName.PORTER_STEMMER)
snowball_df = preprocess.get_tokenlized_df(PreprocessName.SNOWBALL_STEMMER)

# Split datam into train and test
lancaster_model = Model(lancaster_df, preprocess.data['CLASS'])
