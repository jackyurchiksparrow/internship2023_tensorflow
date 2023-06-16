import numpy as np 
import pandas as pd
from keras import layers  
from keras import models
from keras.utils import to_categorical
from keras import optimizers

# read the training file
emnist = pd.read_csv('emnist-balanced-train.csv', header=None)

# getting rid of small letters
emnist = emnist[emnist[0] < 36]
# getting rid of letters O, I, Q,
emnist = emnist[(emnist[0]!=18) & (emnist[0]!=24) & (emnist[0]!=26)]

emnist = np.array(emnist)

# split the data into training and testing, parsing and normalizing ----- start
train_images = emnist[:71280, 1:]
train_images = train_images.reshape((len(train_images), 28, 28, 1))
train_images = train_images.astype('float32') / 255

# categorizing each class
train_labels = to_categorical(emnist[:71280, :1])

test_images = emnist[71280:, 1:]
test_images = test_images.reshape((len(test_images), 28, 28, 1))
test_images = test_images.astype('float32') / 255

test_labels = to_categorical(emnist[71280:, :1])
# split the data into training and testing, parsing and normalizing ----- end

# transposing images returning them to normal condition before training
train_images = np.transpose(train_images, (0, 2, 1, 3))
test_images = np.transpose(test_images, (0, 2, 1, 3))

# CNN model with 36 neurons for each character class to predict
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1), padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(36, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate=1e-4),
                metrics=['accuracy'])

# fit the model
model.fit(train_images, train_labels, epochs=20, batch_size=20, validation_data = (test_images, test_labels))

# save the model
model.save(r'model_letters_numbers_vin.h5')