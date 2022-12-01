import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from preprocess import preprocess_data
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm

data = pd.read_csv("./sam_junk/csv8.csv", low_memory=False)
no_zero_data = data[data.voteBinary != 0]
new_data = preprocess_data(no_zero_data)
tfidf = TfidfVectorizer(analyzer='word')
#
# x_train, x_test, y_train, y_test = train_test_split(new_data['reviewText'], new_data["vote"], test_size=0.3,
#                                                     random_state=42)
#
# x_train = tfidf.fit_transform(x_train)
# x_test = tfidf.transform(x_test)

"""
UNCOMMENT THE LINES BELOW IF YOU WANT TO USE NUMERICAL VALUES
"""

x = no_zero_data.drop(
        ['voteBinary', 'vote', 'verified', 'reviewTime', 'asin', 'style', 'reviewTime', 'reviewText', 'summary', 'image',
         'summary_num_noun', 'summary_num_verb', 'summary_num_adj', 'summary_num_adv', 'summary_num_adp',
         'summary_num_propn', 'summary_num_length', 'preprocess_review'], axis=1, inplace=False)
y = no_zero_data.loc[:, 'vote']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=42)


def random_forest_regressor_preds():
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("Random Forest Regression Training data MAE: " + str(mean_absolute_error(y_test, pred)))
    score = r2_score(y_test, pred)
    print("Random Forest Regression Training data Accuracy: " + str(round(score, 2) * 100))


def logistic_regression():
    model = LogisticRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("Logistic Regression Training data MAE: " + str(mean_absolute_error(y_test, pred)))
    score = r2_score(y_test, pred)
    print("Logistic Regression Training data Accuracy: " + str(round(score, 2) * 100))


def svm_regression_model():
    model = svm.SVR()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("SVM Training data MAE: " + str(mean_absolute_error(y_test, pred)))
    score = r2_score(y_test, pred)
    print("SVM Training data Accuracy: " + str(round(score, 2) * 100))

def linear_regression_model():
    model = svm.LinearSVR()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("Linear Training data MAE: " + str(mean_absolute_error(y_test, pred)))
    score = r2_score(y_test, pred)
    print("Linear Training data Accuracy: " + str(round(score, 2) * 100))

# random_forest_regressor_preds()
# logistic_regression()
# svm_regression_model()
linear_regression_model()
