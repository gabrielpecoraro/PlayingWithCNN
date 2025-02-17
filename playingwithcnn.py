
# %matplotlib inline
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta, Adam, SGD
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Dropout, Flatten, AveragePooling2D, Conv2DTranspose, UpSampling2D, BatchNormalization, RandomFlip, RandomZoom, RandomContrast, RandomBrightness, RandomCrop, RandomTranslation
from keras.models import Sequential
from keras.losses import categorical_crossentropy

"""# **Loading Data**"""

data = np.load('/kaggle/input/mnist-corrnoise-npz/MNIST_CorrNoise.npz')

x_train = data['x_train']
y_train = data['y_train']

num_cls = len(np.unique(y_train))
print('Number of classes: ' + str(num_cls))

print('Example of handwritten digit with correlated noise: \n')

k = 3000
plt.imshow(np.squeeze(x_train[k,:,:]))
plt.show()
print('Class: '+str(y_train[k])+'\n')

# RESHAPE and standarize
x_train = np.expand_dims(x_train/255,axis=3)

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_cls)

print('Shape of x_train: '+str(x_train.shape))
print('Shape of y_train: '+str(y_train.shape))

"""# Training classic CNN"""

model_name='CNN' # To compare models, you can give them different names

pweight='/kaggle/working/weights_' + model_name  + '.keras'

if not os.path.exists('/kaggle/input/weights'):
  print("Does Not exists")
  os.mkdir('./weights')

## EXPLORE VALUES AND FIND A GOOD SET
b_size = 64 # batch size
val_split = 0.2 # percentage of samples used for validation (e.g. 0.5)
ep = 20 # number of epochs

"""## Train the model

# Param for My Model
"""

model_name='CNN' # To compare models, you can give them different names

pweight='/kaggle/working/weights_' + model_name  + '.keras'

if not os.path.exists('/kaggle/input/weights'):
  print("Does Not exists")
  os.mkdir('./weights')

## EXPLORE VALUES AND FIND A GOOD SET
b_size = 128 # batch size
val_split = 0.1 # percentage of samples used for validation (e.g. 0.5)
ep = 120 # number of epochs

# Setting Input
input_shape = x_train.shape[1:4] #(28,28,1)

# Set input shape and number of classes
input_shape = (28, 28, 1)  # Grayscale 28x28 images
num_classes = 10

# Model Definition
model = Sequential()

# Data Augmnetation
model.add(RandomZoom(height_factor=0.1, width_factor=0.1))

# 1st Layer
model.add(Conv2D(64, kernel_size=(3,3), padding="same", input_shape=input_shape,activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

# 2nd Layer
model.add(Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

#3rd Layer
model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Dropout(0.2))

# Flattening Layer
model.add(Flatten())

# Fully Connected Layer
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))


pweight = '/kaggle/working/weights_CNN.keras'
# CheckPointer
checkpointer = ModelCheckpoint(filepath=pweight, verbose=1, save_best_only=True)
# Early Stopping
early_loss = EarlyStopping(monitor = "val_loss", patience = 30, start_from_epoch = 50, mode="min", verbose=1)
# Reduce Learning Rate
reduce_lr = ReduceLROnPlateau(monitor = "val_loss", patience = 7,  mode = "min", verbose=1, factor=0.15)

# Callback List
callbacks_list = [checkpointer, early_loss, reduce_lr]

# Compiling model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

# Fitting model
history=model.fit(x_train, y_train,
                        epochs=ep,
                        batch_size=b_size,
                        verbose=1,
                        shuffle=True,
                        validation_split = val_split,
                        callbacks=callbacks_list)


print('CNN weights saved in ' + pweight)

# Plot loss vs epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

# Plot accuracy vs epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

from keras.models import load_model

## LOAD DATA
data = np.load('/kaggle/input/mnist-corrnoise-npz/MNIST_CorrNoise.npz')

x_test = data['x_test']
y_test = data['y_test']

num_cls = len(np.unique(y_test))
print('Number of classes: ' + str(num_cls))

# RESHAPE and standarize
x_test = np.expand_dims(x_test/255,axis=3)

print('Shape of x_train: '+str(x_test.shape)+'\n')

## Define model parameters
model_name='CNN' # To compare models, you can give them different names
pweight='/kaggle/working/weights_' + model_name  + '.keras'

model = load_model(pweight)
# Instead of predict_classes, use predict and argmax
y_pred_probs = model.predict(x_test)  # Get predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=-1) # Get class with highest probability

Acc_pred = sum(y_pred == y_test)/len(y_test)

print('Accuracy in test set is: '+str(Acc_pred))
