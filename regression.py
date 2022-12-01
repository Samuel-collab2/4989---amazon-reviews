import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from preprocess import preprocess_data
import pandas as pd
import numpy as np

data = pd.read_csv("./sam_junk/csv8.csv", low_memory=False)
new_data = preprocess_data(data)

# x_train, x_test, y_train, y_test = train_test_split(new_data["preprocess_review"], new_data["vote"], test_size=0.3,
#                                                     random_state=42)


def random_forest_regressor_preds():
    x_train, x_test, y_train, y_test = train_test_split(new_data["preprocess_review"], new_data["vote"], test_size=0.3,
                                                        random_state=42)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print("Random Forest Regression Training data MAE: " + str(mean_absolute_error(y_test, pred)))
    errors = abs(pred - y_test)
    print("Random Forest Regression Training data Accuracy: " + str(100 - np.mean(100 * (errors / y_test))))


random_forest_regressor_preds()
