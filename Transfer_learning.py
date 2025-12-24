from keras.backend import dropout, flatten
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense  ,Dropout , GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Define the augmentation parameters
datagen = ImageDataGenerator(
      rescale=1./255,
    rotation_range=20,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest')


datagen2 = ImageDataGenerator(rescale=1./255)

# Load the pre-trained model
model = keras.models.load_model(r"C:\Users\Atef\Desktop\synth model.h5")  # trained model

# Freeze all the layers except the last dense layer in the new model
for layer in model.layers[:-1]:
    layer.trainable = False

# Insert the path of the directory of faces for training and testing
train_generator = datagen.flow_from_directory(
    r'D:\New folder (4)\computer engineering 195339\Level 5\GRADII\Transfer Learning data\utopia\uuu',
    target_size=(125, 125),
    batch_size=32,
    class_mode='categorical'
)

test_generator = datagen2.flow_from_directory(
    r'D:\New folder (4)\computer engineering 195339\Level 5\GRADII\Transfer Learning data\Full data\Test faces',
    target_size=(125, 125),
    batch_size=16,
    class_mode='categorical'
)

train_generator.class_indices

# Get the output of the second to last layer
top_model = model.layers[-2].output

top_model = Dense(1024, activation='relu', name='dense____1')(top_model)
top_model = Dense(1024, activation='relu', name='dense____2')(top_model)
top_model = Dense(512, activation='relu', name='dense____3')(top_model)
top_model = Dense(train_generator.num_classes, activation='softmax', name='dense_output')(top_model)

# Create the new model
new_model = keras.Model(inputs=model.input, outputs=top_model)

print (new_model.summary())

checkpoint = ModelCheckpoint (r"C:\Users\Atef\Desktop\recognition final 7-6-2023 94%(Utopia Last day)5.h5", monitor="loss", mode="min", save_best_only =True, verbose=1)
earlystop =  EarlyStopping (monitor= 'loss', min_delta = 0, patience = 15, verbose= 1, restore_best_weights = True)
callbacks = [checkpoint] 

# Compile the new model with a lower learning rate
optimizer = keras.optimizers.Adam(lr=0.001)

new_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Fine-tune the model
new_model.fit_generator(    
    train_generator,
    epochs=500,
    callbacks=callbacks

)

# Evaluate the model's accuracy on the validation data
#_, val_acc = new_model.evaluate(test_generator)
#print('Validation accuracy:', val_acc)

# Saving the model
new_model.save(r'C:\Users\Atef\Desktop\recognition final 7-6-2023 94%(Utopia Last day)5.h5')
