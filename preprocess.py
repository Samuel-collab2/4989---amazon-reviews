import pandas as pd
import string
import re
from keras_preprocessing.text import text_to_word_sequence
import nltk
from sklearn.model_selection import train_test_split

nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("./sam_junk/csv8.csv", low_memory=False)

nona = data.dropna(axis=0, how="any")

overall = data.loc[:, "overall"]
review_text = data.loc[:, "reviewText"]
summary = data.loc[:, "summary"]
vote = data.loc[:, "vote"]
verified = data.loc[:, "verified"]


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


def tokenization(text):
    tokens = re.split('W+', text)
    return tokens


def remove_stopwords(text):
    stopwords_list = stopwords.words('english')
    output = [i for i in text if i not in stopwords_list]
    return output


def lemmatizer(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

def preprocess_data(raw_data):
    preprocess_reviews = raw_data
    preprocess_reviews['preprocess_review'] = raw_data['reviewText'].apply(lambda x: remove_punctuation(x))
    preprocess_reviews['preprocess_review'] = preprocess_reviews['preprocess_review'].apply(lambda x: x.lower())
    preprocess_reviews['preprocess_review'] = preprocess_reviews['preprocess_review'].apply(lambda x: text_to_word_sequence(x))
    preprocess_reviews['preprocess_review'] = preprocess_reviews['preprocess_review'].apply(lambda x: remove_stopwords(x))
    preprocess_reviews['preprocess_review'] = preprocess_reviews['preprocess_review'].apply(lambda x: lemmatizer(x))

    for index, i in enumerate(preprocess_reviews['preprocess_review']):
        if len(i) == 0:
            print("Found empty list in: " + str(index))
            preprocess_reviews.drop([index], axis=0, inplace=True)
        if len(i) == 1 and len(i[0]) == 1:
            print("Found single letter in: " + str(index))
            preprocess_reviews.drop([index], axis=0, inplace=True)
        preprocess_reviews['preprocess_review'][index] = " ".join(i)

    # bag_of_words_values = []
    # for index, v in enumerate(preprocess_reviews['preprocess_review']):
    #     bag_of_words_values.append(bag_of_words(v))
    # preprocess_reviews['preprocess_review'] = bag_of_words_values


    return preprocess_reviews

def bag_of_words(text_data):
    vectorizer = CountVectorizer(analyzer='word',ngram_range=(2,2))
    X = vectorizer.fit_transform(text_data)
    return X.toarray()


# preprocess_data = preprocess_data(data)
# print(preprocess_data)
# print("hello")
# X_train, X_test, y_train, y_test = train_test_split(data["preprocess_review"], data["vote"], test_size = 0.3)
# print("wasup")
# print(X_train)
# print(y_train)
# for index, i in enumerate(X_train):
#     if len(i) != 0:
#         X_train[index] = bag_of_words(i, index)
# print("bye")
# print(X_train[19929])

