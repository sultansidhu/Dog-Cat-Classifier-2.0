'''This file is a basic convolutional neural network for classifying dogs and cats. Various optimization
   techniques have been used to bolster accuracy in the model.
'''
# IMPORTS
import os
import shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

# MAKING DIRECTORIES AND MOVING FILES
original_dataset = '/Users/sultansidhu/Desktop/development/python/ai/classifier(dogs vs cats)/dogs-vs-cats'
directories = []

dogs_train_set = os.path.join(original_dataset, 'dogs_train_set')
os.mkdir(dogs_train_set)
directories.append(dogs_train_set)
dogs_validation_set = os.path.join(original_dataset, 'dogs_validation_set')
os.mkdir(dogs_validation_set)
directories.append(dogs_validation_set)
dogs_test_set = os.path.join(original_dataset, 'dogs_test_set')
os.mkdir(dogs_test_set)
directories.append(dogs_test_set)

cats_train_set = os.path.join(original_dataset, 'cats_train_set')
os.mkdir(cats_train_set)
directories.append(cats_train_set)
cats_validation_set = os.path.join(original_dataset, 'cats_validation_set')
os.mkdir(cats_validation_set)
directories.append(cats_validation_set)
cats_test_set = os.path.join(original_dataset, 'cats_test_set')
os.mkdir(cats_test_set)
directories.append(cats_test_set)

train_src = os.path.join(original_dataset, 'train')

filenames = ['cat.{}.jpg'.format(i) for i in range(0, 4000)]
for file in filenames:
    src = os.path.join(train_src, file)
    dst = os.path.join(cats_train_set, file)
    shutil.copyfile(src, dst)

filenames = ['cat.{}.jpg'.format(i) for i in range(4000, 5000)]
for file in filenames:
    src = os.path.join(train_src, file)
    dst = os.path.join(cats_validation_set, file)
    shutil.copyfile(src, dst)

filenames = ['cat.{}.jpg'.format(i) for i in range(5000, 6000)]
for file in filenames:
    src = os.path.join(train_src, file)
    dst = os.path.join(cats_test_set, file)
    shutil.copyfile(src, dst)

filenames = ['dog.{}.jpg'.format(i) for i in range(0, 4000)]
for file in filenames:
    src = os.path.join(train_src, file)
    dst = os.path.join(dogs_train_set, file)
    shutil.copyfile(src, dst)

filenames = ['dog.{}.jpg'.format(i) for i in range(4000, 5000)]
for file in filenames:
    src = os.path.join(train_src, file)
    dst = os.path.join(dogs_validation_set, file)
    shutil.copyfile(src, dst)

filenames = ['dog.{}.jpg'.format(i) for i in range(5000, 6000)]
for file in filenames:
    src = os.path.join(train_src, file)
    dst = os.path.join(dogs_test_set, file)
    shutil.copyfile(src, dst)

# MODEL CONSTRUCTION

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# DATA PREPROCESSING

dogs_train_datagen = ImageDataGenerator(rescale=1./255)
cats_train_datagen = ImageDataGenerator(rescale=1./255)
dogs_test_datagen = ImageDataGenerator(rescale=1./255)
cats_test_datagen = ImageDataGenerator(rescale=1./255)

cats_train_generator = cats_train_datagen.flow_from_directory(cats_train_set, target_size=(150, 150),
                                                             batch_size=20, class_mode='binary')
dogs_train_generator = dogs_train_datagen.flow_from_directory(dogs_train_set, target_size=(150, 150),
                                                             batch_size=20, class_mode='binary')
cats_validation_generator = cats_test_datagen.flow_from_directory(cats_validation_set, target_size=(150, 150),
                                                             batch_size=20, class_mode='binary')
dogs_validation_generator = dogs_test_datagen.flow_from_directory(dogs_validation_set, target_size=(150, 150),
                                                             batch_size=20, class_mode='binary')

