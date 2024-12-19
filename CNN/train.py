import pickle
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback

dimx = 220
dimy = 220
filepath = "saved-model_MPIIy1.keras"

# Load label dictionary
with open('labelsdict_mpii', 'rb') as infile:
    labels = pickle.load(infile)

# Get list of training and validation images
X_train = [i for i in os.listdir('./train/') if i.endswith('.jpg')]
X_valid = [i for i in os.listdir('./valid/') if i.endswith('.jpg')]

class DataGenerator(Sequence):
    def __init__(self, list_IDs, labels, batch_size, dim, n_channels, path_X, shuffle=False):
        self.dim = dim                                               
        self.batch_size = batch_size                                   
        self.labels = labels                                         
        self.list_IDs = list_IDs                                     
        self.n_channels = n_channels                                 
        self.shuffle = shuffle                        
        self.path_X = path_X                            
        self.on_epoch_end()                           

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y = self.__itercustom__(list_IDs_temp)
        X = X / 255.0
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __itercustom__(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, 16 * 2))
        for i, ID in enumerate(list_IDs_temp):
            X[i,:,:,:] = cv2.imread(os.path.join(self.path_X, ID), 1)
            Y[i,:] = np.append(labels[ID]['x'], labels[ID]['y'], axis=0)
        return X, Y
    
# Initialize generators
path_to_X_train = './train/'
path_to_X_valid = './valid/'
batch_size_train = 128
batch_size_valid = 128

training_generator = DataGenerator(list_IDs=X_train, labels=labels, 
                                   batch_size=batch_size_train, dim=(dimy, dimx), 
                                   n_channels=3, path_X=path_to_X_train, 
                                   shuffle=True)

validation_generator = DataGenerator(list_IDs=X_valid, labels=labels, 
                                     batch_size=batch_size_valid, dim=(dimy, dimx), 
                                     n_channels=3, path_X=path_to_X_valid, 
                                     shuffle=True)

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 30:
        return 0.0005
    else:
        return 0.0005 * np.exp(0.1 * (30 - epoch))

# Model checkpoint
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                             save_best_only=True, save_weights_only=False)

# Custom callback to print learning rate at the beginning of each epoch
class PrintLearningRate(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        print(f'Learning Rate for Epoch {epoch+1}: {lr}')

# Define the lr_schedule variable
lr_schedule = LearningRateScheduler(scheduler)

# Define the callback to print learning rate
print_lr = PrintLearningRate()

# Model definition
model = Sequential()
model.add(DepthwiseConv2D(kernel_size=(1,1), strides=(4, 4), depth_multiplier=1,
                          use_bias=False, input_shape=(220,220,3)))
model.add(Conv2D(96, (11, 11), activation='relu', padding='same'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(16*2, activation='tanh'))
    
model.compile(loss=tf.losses.MeanSquaredError(), optimizer=Adam(), metrics=[tf.metrics.MeanAbsoluteError()])

# Model summary
model.summary()

# Model training
model.fit(x=training_generator, validation_data=validation_generator, 
          epochs=20, verbose=1, callbacks=[checkpoint, lr_schedule, print_lr])
