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
import matplotlib.pyplot as plt

# MAKING DIRECTORIES AND MOVING FILES
original_dataset = '/Users/sultansidhu/Desktop/development/python/ai/classifier(dogs vs cats)/dogs-vs-cats'
directories = []

# making sub directories for training, validation and testing
master_train_path = os.path.join(original_dataset, 'master_train')
master_validation_path = os.path.join(original_dataset, 'master_validation')
master_test_path = os.path.join(original_dataset, 'master_test')
try:
    os.mkdir(master_train_path)
    os.mkdir(master_validation_path)
    os.mkdir(master_test_path)
except FileExistsError:
    print("File exists, moving on...")

# making the dog directories
dogs_train_set = os.path.join(master_train_path, 'dogs_train_set')
directories.append(dogs_train_set)
dogs_validation_set = os.path.join(master_validation_path, 'dogs_validation_set')
directories.append(dogs_validation_set)
dogs_test_set = os.path.join(master_test_path, 'dogs_test_set')
directories.append(dogs_test_set)
try:
    os.mkdir(dogs_train_set)
    os.mkdir(dogs_validation_set)
    os.mkdir(dogs_test_set)
except FileExistsError:
    print("File exists, moving on...")

# making the cat directories
cats_train_set = os.path.join(master_train_path, 'cats_train_set')
directories.append(cats_train_set)
cats_validation_set = os.path.join(master_validation_path, 'cats_validation_set')
directories.append(cats_validation_set)
cats_test_set = os.path.join(master_test_path, 'cats_test_set')
directories.append(cats_test_set)
try:
    os.mkdir(cats_train_set)
    os.mkdir(cats_validation_set)
    os.mkdir(cats_test_set)
except FileExistsError:
    print("File exists, moving on...")

# file transfer
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
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# DATA PREPROCESSING

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, )

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(master_train_path, target_size=(150, 150),
                                                    batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(master_validation_path, target_size=(150, 150),
                                                        batch_size=32, class_mode='binary')
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=validation_generator, validation_steps=50)

model.save('cats_vs_dogs_small_v2.h5')


if __name__ == "__main__":
    # PLOTTING GRAPH DYNAMICS

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, 1 + len(acc))

    plt.plot(epochs, acc, 'bo', label='Training Acc')
    plt.plot(epochs, val_acc, 'b', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()
