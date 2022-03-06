def predict_bank(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import stats
    import joblib

    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import preprocessing
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    pipe = joblib.load('model.pkl')
    df_temp = df.drop('y', axis=1)
    pred = pipe.predict_proba(df_temp)
    positive_class = []
    for i in pred:
        positive_class.append(i[1])
    df_pred = pd.DataFrame(data=positive_class, columns=['make_deposit'])
    df_pred = pd.concat([df.loc[:, ['age', 'job', 'marital', 'education', 'y']], df_pred], axis=1)
    return df_pred

def get_output_schema():
     return pd.DataFrame({
    'age' : prep_int(),
    'job' : prep_string(),
    'marital' : prep_string(),
    'education' : prep_string(),
    'y': prep_string(),
    'make_deposit' : prep_decimal()
     })