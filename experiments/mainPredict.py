# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model

from helpers.helpers import z_score, fill_missing, split_dataset
from models.models import diabetic_status_predictor, BatchGenerator
from models.hparams import *

# Load dataset
data_df = pd.read_csv('../dataset/diabetes.csv')

# Fill in missing values
if FILL_DATA:
    fill_missing(data_df)

# Re-Scale input features and separate input/output vectors
if RESCALE_DATA:
    features = z_score(np.array(data_df)[:, FEATURE_COLUMNS])
else:
    features = np.array(data_df)[:, FEATURE_COLUMNS]
n_features = features.shape[1]
status = np.array(data_df['Outcome'])

# Split data into train-validation-test
train_x, valid_x, train_y, valid_y, test_x, test_y = split_dataset(features, status)

# Define model
if (not FILL_DATA) & (not RESCALE_DATA):
    model_path = '../pretrained/predictor_notFilled_notScaled.h5'
elif (not FILL_DATA) & RESCALE_DATA:
    model_path = '../pretrained/predictor_notFilled_scaled.h5'
elif FILL_DATA & (not RESCALE_DATA):
    model_path = '../pretrained/predictor_filled_notScaled.h5'
else:
    model_path = '../pretrained/predictor_filled_scaled.h5'

checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', mode='min', verbose=0,
                             save_best_only=True, save_weights_only=False)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', mode='min', verbose=0, factor=0.9, min_delta=0.0001, patience=10)
earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=20)
callback_list = [checkpoint, reduceLROnPlat, earlyStop]

train_generator = BatchGenerator(train_x, train_y, batch_size=BATCH_SIZE, is_train=True)
valid_generator = BatchGenerator(valid_x, valid_y, batch_size=BATCH_SIZE, is_train=False)

# LOAD/COMPILE MODEL
if MODEL_MODE == 'USE_PRE_TRAINED' and os.path.isfile(model_path):
    model = load_model(model_path)
elif MODEL_MODE == 'RESUME_TRAINING' and os.path.isfile(model_path):
    model = load_model(model_path)
    # TRAIN
    model.fit(
        train_generator,
        steps_per_epoch=np.ceil(float(len(train_x)) / float(BATCH_SIZE)),
        validation_data=valid_generator,
        validation_steps=np.ceil(float(len(valid_x)) / float(BATCH_SIZE)),
        epochs=N_EPOCHS,
        verbose=0,
        callbacks=callback_list)
elif MODEL_MODE == 'RESET_MODEL' or not os.path.isfile(model_path):
    model_name = 'Diabetic_Predictor'
    model = diabetic_status_predictor(model_name, input_shape=(n_features,))
    model.compile(optimizer=Adam(lr=1e-3), loss='mse', metrics=['acc'])
    # TRAIN
    model.fit(
        train_generator,
        steps_per_epoch=np.ceil(float(len(train_x)) / float(BATCH_SIZE)),
        validation_data=valid_generator,
        validation_steps=np.ceil(float(len(valid_x)) / float(BATCH_SIZE)),
        epochs=N_EPOCHS,
        verbose=0,
        callbacks=callback_list
    )
else:
    raise NameError('Un-specified mode for model!')

# Prediction
raw_predictions = model.predict(test_x).reshape((test_x.shape[0],))
binary_predictions = np.zeros_like(raw_predictions)
binary_predictions[raw_predictions>0.5] = 1

# Accuracy
accur = np.sum(binary_predictions==test_y)/test_x.shape[0]
print(f'Model Accuracy: {accur:0.3f}')
