# Helper functions for decision tree. As we aren't allowed to use 
# sklearn's train_test_split(), we can create our own easily
import numpy as np
import pandas as pd

def train_test_split(df, test_size=0.2):

    '''Splits a dataframe into training and testing data'''

    # account for floats
    test_size = round((test_size) * len(df))

    test_indices = np.random.default_rng().choice(a=df.index.to_list(), size=test_size)

    test_data = df.loc[test_indices]

    train_data = df.drop(test_indices)

    return train_data, test_data

# Notes:
    # test size could return a fraction of a index so we round it first
    # ex. len(df) = 93
    # 18.6 = 0.2 * 93
    # 19 = round(18.6)

# Use of np.random.Generator.choice() rather than np.random.sample()
# https://stackoverflow.com/questions/40914862/why-is-random-sample-faster-than-numpys-random-choice

if __name__ == "__main__":
     
    print('train:', len(train))
    print('test:', len(test))