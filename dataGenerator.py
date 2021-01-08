import os
from natsort import natsorted
import pandas as pd
from itertools import combinations
from collections import namedtuple

Experiment = namedtuple('Experiment', ['trainMonkeys', 'testMonkey', 
                                       'trainX', 'trainY', 'testX', 'testY', 'slice'])
                                       
def initialize_database(data_path):
    db = []
    
    # for each slice
    for sliceId in os.listdir(data_path):
        wd = os.path.join(data_path, sliceId)
        
        # for each sample in each slice
        for sample in os.listdir(wd):
            
            # extract sample metadata
            tokens = sample.split('_')
            monkeyId = tokens[0]
            week = tokens[1]
            
            # build the relevant database entry
            db.append({
                'monkeyId': monkeyId,
                'slice': sliceId,
                'label': week,
                'path': os.path.join(wd, sample)
            })
            
    # return a pandas dataframe
    return pd.DataFrame(db)
    
def get_data(trainMonkeys, testMonkey, sliceId, database):
    # select relevant slice from database
    sliceData = database[database['slice'] == str(sliceId)]

    # select the trainMonkey and testMonkey data
    train = sliceData[sliceData['monkeyId'].isin(trainMonkeys)]
    test = sliceData[sliceData['monkeyId'] == testMonkey]
    
    # return training and testing data
    return train['path'], train['label'], test['path'], test['label']

def generate_experiment_data(dataset_path, slices=None):
    # create a database
    database = initialize_database(dataset_path)
    
    # use all slices if not specified
    if slices is None:
        slices = natsorted(database['slice'].unique())
    # get list of all monkeys from database
    monkeys = natsorted(database['monkeyId'].sort_values().unique())
    
    for sliceId in slices:
        # build combinations of experiments (6C5 = 6 experiments)
        
        for trainMonkeys in combinations(monkeys, 5):
            # the monkey not in the combination is the test monkey
            testMonkey = [i for i in monkeys if i not in trainMonkeys][0]
            
            # return Experiment tuple
            yield Experiment(
                trainMonkeys,
                testMonkey,
                *(get_data(trainMonkeys, testMonkey, sliceId, database)),
                sliceId
            )
            

