import os
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from random_brain import random_brain

brain = random_brain.random_brain()

def preprocess(df):
    # convert targets into 1 or 0
    # No, Negative, Male = 0
    # Yes, Positive, Female = 1
    df = df.replace(to_replace =["No", "Negative", 'Male'], value =0)
    df = df.replace(to_replace =["Yes", "Positive", 'Female'], value =1)


    data=df.iloc[:,:-1].to_numpy()
    data_targets=df.iloc[:,-1]
    data_targets=np_utils.to_categorical(data_targets, num_classes=2)

    return data, data_targets

def run_test():
    # import models to brain
    import_path = os.path.join(os.getcwd() + '/models')
    brain.import_models(model_path=import_path)
    brain.show_brain()
    
    # import and preprocess data
    data_path = os.path.join(os.getcwd() + '/data/diabetes.csv')
    df = pd.read_csv(data_path)
    xdata, ydata = preprocess(df)
    
    #work on predictions
    votes = brain.vote(xdata)
    votes = votes.mean(axis=0)

    prediction = np.where(votes > 0.5, 1, 0)

    # generate report
    report = classification_report(ydata, prediction)
    print(report)


# entry point
if __name__ == '__main__':
    run_test()