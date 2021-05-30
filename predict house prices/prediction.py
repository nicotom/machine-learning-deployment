import pandas as pd
import numpy as np
from joblib import load

model = load('rf.joblib')


def predict(dict_values):
    return model.predict(pd.DataFrame([dict_values]))[0]
