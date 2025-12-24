#############################################
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

datagen2 = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

###################################################################################################
import cv2
train_generator = datagen.flow_from_directory(
    r'D:\Downloads Folder\part1 72 digiface - Copy',
    target_size=(125, 125),
    batch_size=128,
    class_mode='categorical')
train_generator.class_indices

test_generator = datagen2.flow_from_directory(
    r'D:\Downloads Folder\test 72',
    target_size=(125, 125),
    batch_size=32,
    class_mode='categorical')

###################################################################################################
from keras.models import Sequential, load_model 
from keras.layers import regularization
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense , BatchNormalization ,Dropout

if K.image_data_format() == 'channels_first':
    input_shape = (3, 125, 125)
else:
    input_shape = (125, 125, 3)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(125, 125,3)))

# Add additional convolutional layers
model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=150, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten the output of the convolutional layers
model.add(Flatten())

# Add fully connected layers for classification
model.add(Dense(units=256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(units=2400 , activation='softmax'))



# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=200,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size)


model=load_model(r"C:\Users\Atef\Desktop\recognition model with 67167 identity 16-5-2023.h5")
_, val_acc = model.evaluate(test_generator)
print('Validation accuracy:', val_acc)


#saving the model
#model.save(r'C:\Users\Atef\Desktop\recognition model with 67167 identity 16-5-2023.h5')
