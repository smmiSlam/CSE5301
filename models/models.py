# Logistic Regression model
import numpy as np
from sklearn.utils import shuffle
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout, Dense
from keras.utils import Sequence

from models.hparams import RAND_SEED, HIDDEN_UNITS


# Diabetic status prediction model
def diabetic_status_predictor(model_name, input_shape):
    # Input layer
    input_tensor = Input(shape=input_shape, name='input0')
    # First hidden layer
    x = Dense(HIDDEN_UNITS[0], name='fc0')(input_tensor)
    x = BatchNormalization(name='bn0')(x)
    x = Activation('relu', name='a0')(x)
    x = Dropout(0.1)(x)
    # Second hidden layer
    x = Dense(HIDDEN_UNITS[1], name='fc1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='a1')(x)
    x = Dropout(0.1)(x)
    # Output layer
    x = Dense(HIDDEN_UNITS[2], activation='sigmoid', name='fc2')(x)
    # Define and return model
    model = Model(inputs=input_tensor, outputs=x, name=model_name)
    return model


# Batch generator
class BatchGenerator(Sequence):
    def __init__(self, features, status, batch_size, is_train=False):
        self.features = features
        self.status = status
        self.batch_size = batch_size
        self.is_train = is_train
        if self.is_train:
            self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.status)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.features[idx*self.batch_size: (idx+1)*self.batch_size]
        batch_y = self.status[idx*self.batch_size: (idx+1)*self.batch_size]
        return self.train_generate(batch_x, batch_y) \
            if self.is_train else self.valid_generate(batch_x, batch_y)

    def on_epoch_end(self):
        self.features, self.status = shuffle(self.features, self.status, random_state=RAND_SEED)

    def train_generate(self, batch_x, batch_y):
        return batch_x, batch_y

    def valid_generate(self, batch_x, batch_y):
        return batch_x, batch_y
