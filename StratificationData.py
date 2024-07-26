from sklearn.model_selection import StratifiedShuffleSplit

def StratifiedSampling(housing_data, target_column, test_size=0.2, random_state=42):
    split = StratifiedShuffleSplit(n_splits = 1, test_size = test_size, random_state = random_state)
    for train_index, test_index in split.split(housing_data, housing_data[target_column]):
        train_set = housing_data.loc[train_index]
        test_set = housing_data.loc[test_index]
    
    return train_set, test_set