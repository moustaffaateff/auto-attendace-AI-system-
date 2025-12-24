
from numba import jit, cuda
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)
datagen2 = ImageDataGenerator(rescale=1./255)

    ###################################################################################################
import cv2
train_generator = datagen.flow_from_directory(
        r'D:\train',
        target_size=(125, 125),
        batch_size=64,
        class_mode='categorical')
train_generator.class_indices

test_generator = datagen2.flow_from_directory(
        r'D:\new path',
        target_size=(125, 125),
        batch_size=32,
        class_mode='categorical')
@jit(target='cuda')
def train_on_gpu(train_generator):
   

    ###################################################################################################
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", input_shape=(125, 125,3)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))


    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(150, activation='softmax'))


    # Compile the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=32,
        epochs=100
    )

    test_loss, test_acc = model.evaluate_generator(train_generator)
    print('Test accuracy:', train_generator)
    #saving the model
    model.save(r'C:\Users\Atef\Desktop\recognition.h5')
train_on_gpu(train_generator)