# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from models.hparams import RAND_SEED, PORTION_TEST, PORTION_VALID


# Function to z-score the features
def z_score(data):
    scaled_features = list()
    for col_idx in range(data.shape[1]):
        features = data[:, col_idx]
        features -= features.min()
        features = features/features.max()
        scaled_features.append(features)
    return np.array(scaled_features).T


# Fill in missing values from the input features
def fill_missing(data_df):
    zero_value_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in zero_value_columns:
        feature_values_with_outcome_0 = data_df[column][(data_df['Outcome'].values==0) & (data_df[column].values != 0)].values
        median_value_with_outcome_0 = np.mean(feature_values_with_outcome_0)
        zero_val_locs_with_outcome_0 = (data_df[column].values == 0) & (data_df['Outcome'].values == 0)
        data_df[column][zero_val_locs_with_outcome_0] = median_value_with_outcome_0

        feature_values_with_outcome_1 = data_df[column][(data_df['Outcome'].values==1) & (data_df[column].values != 0)].values
        median_value_with_outcome_1 = np.mean(feature_values_with_outcome_1)
        zero_val_locs_with_outcome_1 = (data_df[column].values == 0) & (data_df['Outcome'].values == 1)
        data_df[column][zero_val_locs_with_outcome_1] = median_value_with_outcome_1


# Split dataset into train, test, validation
def split_dataset(features, status):
    train_valid_x, test_x, train_valid_y, test_y = train_test_split(features, status, test_size=PORTION_TEST,
                                                                    random_state=RAND_SEED)
    train_x, valid_x, train_y, valid_y = train_test_split(train_valid_x, train_valid_y, test_size=PORTION_VALID,
                                                          random_state=RAND_SEED)
    return train_x, valid_x, train_y, valid_y, test_x, test_y
