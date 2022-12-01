import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv("./sam_junk/csv8.csv", low_memory=False)
no_zero_data = data[data.voteBinary != 0]


def predict_claim_amount_NN():

    x = no_zero_data.drop(
        ['voteBinary', 'vote', 'verified', 'reviewTime', 'asin', 'style', 'reviewTime', 'reviewText', 'summary', 'image',
         'summary_num_noun', 'summary_num_verb', 'summary_num_adj', 'summary_num_adv', 'summary_num_adp',
         'summary_num_propn', 'summary_num_length'], axis=1, inplace=False)
    y = no_zero_data.loc[:, 'vote']
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(x)
    print(X_scale)
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    number_of_features = x_train.shape[1]

    model = Sequential()
    model.add(Input(shape=(number_of_features,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error',
                  metrics=['accuracy'])

    # fit the model to training data
    model.fit(x_train, y_train, epochs=100)
    pred = model.predict(x_test)

    mae = np.mean(abs(pred - y_test.to_numpy()))
    print("Training data MAE: " + str(mae))
    print("predictions: " + str(pred))

    score = r2_score(y_test, pred)
    print("NN Training data Accuracy: " + str(round(score, 2) * 100))
    return score

number_of_runs = 5
avg_score = []
for x in range(number_of_runs):
    avg_score.append(predict_claim_amount_NN())

sum = 0
for y in avg_score:
    sum += y
print(sum/len(avg_score))