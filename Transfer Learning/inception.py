####
# Transfer learning using Inception pretrained model for CIFAR-10 classification.
# Best Model: 71.23% validation accuracy in 10 epochs.
####
import cv2
import numpy as np
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.datasets import cifar10
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras import backend as K

#
# Model Configuration
#
num_classes = 10
batch_size = 32
epochs = 10
img_width, img_height = 150, 150

# Load and split data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Resize dataset
print('Resizing dataset...')
inception_x_train = np.array([cv2.resize(image, (img_width, img_height)) for image in x_train])
inception_x_test = np.array([cv2.resize(image, (img_width, img_height)) for image in x_test])

# Preprocess data
print('Preprocessing dataset...')
inception_x_train = inception_x_train.astype('float32')
inception_x_test = inception_x_test.astype('float32')
inception_x_train /= 255
inception_x_test /= 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#
# CNN Structure
#
base_model = InceptionV3(weights='imagenet', 
	                     include_top=False)

# Global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# FC
x = Dense(64, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', 
	          loss='categorical_crossentropy',
	          metrics = ['accuracy'])

# Fit image generator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)   # randomly flip images

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
print 'Fitting image generator...'
datagen.fit(inception_x_train)

# Display model parameters and structure
print(model.summary())

# Train the model
model.fit_generator(datagen.flow(inception_x_train, y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(inception_x_test, y_test),
                        workers=4)

# Score trained model
scores = model.evaluate(inception_x_test, y_test, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])