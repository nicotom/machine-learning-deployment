import pandas as pd
from joblib import load

model = load('rf.joblib')


def predict(dict_values):
    return model.predict(pd.DataFrame([dict_values]))[0]
