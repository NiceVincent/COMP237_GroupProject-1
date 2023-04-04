
import neurolab as nl
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from Model import Model
import neurolab as nl

from Preprocess import Preprocess, PreprocessName

RANDOMS_SEED = 420
TEST_SIZE = 0.4

# Preprocess
preprocess = Preprocess()
lancaster_df = preprocess.get_tokenlized_df(PreprocessName.LANCASTER_STEMMER)
# lemmatize_df = preprocess.get_tokenlized_df(PreprocessName.LEMMATIZE)
# porter_df = preprocess.get_tokenlized_df(PreprocessName.PORTER_STEMMER)
# snowball_df = preprocess.get_tokenlized_df(PreprocessName.SNOWBALL_STEMMER)

print(preprocess.data['CLASS'])
print(lancaster_df)
# Split datam into train and test
model = Model(lancaster_df, preprocess.data['CLASS'])
print("=======")
print(model.x_train)
print("=======")
print(model.y_train)
nn_ex1 = nl.net.newff(model.min_max_pair(), [100, 50, 25])
error_progress_ex1 = nn_ex1.train(model.x_train, model.y_train,
                                  epochs=1000, show=15, goal=0.00001)

print(error_progress_ex1)
