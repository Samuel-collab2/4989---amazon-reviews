import numpy as np
import pandas as pd
import json
import random
import re
import spacy
# import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from tensorflow import keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import string


def wordCloud_generator(data, title=None):
    wordcloud = WordCloud(width=800, height=800,
                          background_color='black',
                          min_font_size=10
                          ).generate(" ".join(data.values))
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(title, fontsize=30)
    plt.show()


def cleanData(reviews):
    all_ = []

    data = open("../apostrophe_contractions.json", "r")
    appos = json.load(data)

    data = open("../apostrophe_typos.json", "r")
    apposV2 = json.load(data)

    for review in reviews:
        lower_case = review.lower()  # lower case the text
        lower_case = lower_case.replace(" n't", " not")  # correct n't as not
        lower_case = lower_case.replace(".", " . ")
        lower_case = ' '.join(word.strip(string.punctuation) for word in lower_case.split())  # remove punctuation

        words = lower_case.split()  # split into words
        words = [word for word in words if word.isalpha()]  # remove numbers

        split = [apposV2[word] if word in apposV2 else word for word in
                 words]  # correct using apposV2 as mentioned above
        split = [appos[word] if word in appos else word for word in split]  # correct using appos as mentioned above
        split = [word for word in split if word not in stop]  # remove stop words

        reformed = " ".join(split)  # join words back to the text
        doc = nlp(reformed)
        reformed = " ".join([token.lemma_ for token in doc])  # lemmatiztion
        all_.append(reformed)

    df_cleaned = pd.DataFrame()
    df_cleaned['clean_reviews'] = all_
    return df_cleaned['clean_reviews']


def train_test_split(cleaned_data, labels, y_ohe, train_ratio=0.75):
    """
    Splits the inputted cleaned data features and labels into train and test datasets
    :param cleaned_data: a Pandas DataFrame, data features
    :param labels: a Pandas Dataframe, data labels
    :param train_ratio: training-to-test ratio, 75% by default
    :return: 4 Pandas DataFrames
    """
    num_rows = cleaned_data.shape[0]

    shuffled_indices = list(range(num_rows))
    random.seed(42)
    random.shuffle(shuffled_indices)

    train_set_size = int(num_rows * train_ratio)

    train_indices = shuffled_indices[:train_set_size]
    test_indices = shuffled_indices[train_set_size:num_rows]

    train_features = cleaned_data.iloc[train_indices]
    train_labels = labels.iloc[train_indices]

    test_features = cleaned_data.iloc[test_indices]
    test_labels = labels.iloc[test_indices]

    return train_features, train_labels, test_features, test_labels


if __name__ == '__main__':
    """
    Tutorial Link:
    https://medium.com/analytics-vidhya/predicting-the-ratings-of-reviews-of-a-hotel-using-machine-learning-bd756e6a9b9b
    """

    data = pd.read_csv('/Users/at/Downloads/tripadvisor_hotel_reviews.csv')
    X = data['Review'].copy()
    y = data['Rating'].copy()


    """Pre-processing"""
    # nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    # # nltk.download('stopwords')
    # stop = stopwords.words('english')
    #
    # X_cleaned = cleanData(X)
    # X_cleaned.head()
    # # X_cleaned.to_csv("./cleaned_data.csv")  # saves cleaned data to CSV so don't have to clean everytime

    X_cleaned = pd.read_csv("./cleaned_data.csv")
    X_cleaned = X_cleaned.iloc[:, 1]

    """
    Encode Target Variable Rating
    """
    encoding = {1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4
                }
    labels = ['1', '2', '3', '4', '5']

    y_ohe = data['Rating'].copy()
    y_ohe.replace(encoding, inplace=True)
    y_ohe = to_categorical(y_ohe, 5)

    """
    Split into 80% for train, 10% for CV, 10% for testing
    """
    X_train, y_train, X_test, y_test = train_test_split(X_cleaned, y, y_ohe)

    """
    Sequentializing Data
    """
    from keras.preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)

    max_length = max([len(x) for x in X_train])
    vocab_size = len(tokenizer.word_index) + 1

    print(f"Vocabulary size: {vocab_size}")
    print(f"Max length of sentence: {max_length}")

    X_train = pad_sequences(X_train, max_length, padding='post')

    """
    Creating the Model
    """
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.layers import Bidirectional, Embedding, Flatten
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    embedding_vector_length = 32
    num_classes = 5
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=X_train.shape[1]))
    model.add(Bidirectional(LSTM(250, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                 ModelCheckpoint('../model/model.h5', save_best_only=True,
                                 save_weights_only=False)]
    model.summary()

    """
    Fitting Model & Starting Training
    """
    history = model.fit(X_train, y_train, validation_split=0.11,
                        epochs=15, batch_size=32, verbose=1,
                        callbacks=callbacks)

    pass
"""
https://stackoverflow.com/questions/52677634/pycharm-cant-find-spacy-model-en

"""