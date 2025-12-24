# Used Libraries 
from mtcnn import MTCNN
import cv2
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import keras.utils as image
import cv2
from keras.models import load_model
from PIL import Image 
from keras.preprocessing.image import ImageDataGenerator
import os
import glob



#------------------------------------------------------------------------------------------------------------#
def webcam_connection ():
    webCam = cv2.VideoCapture(0)
    currentframe = 1
    path = r"C:\Users\Atef\Desktop\computer engineering 195339\GRADII\line 54" #"" search for how to name the folders in the raspberry pi
    while (currentframe<31):
        success, frame = webCam.read()

        # Save Frame by Frame into disk using imwrite method
   
        cv2.imwrite(os.path.join(path,'Frame' + str(currentframe) + '.jpg'), frame) #save each captuerd frame in frames folder 
        currentframe += 1  # Inc the number of saved farmes to keep track of the number of frames
        time.sleep(30)  # Wait 30 second between 2 frames 


    # At this stage 30 frames are captured and saved to be processed 

    # Draw the face box 
#------------------------------------------------------------------------------------------------------------#   
def detection_phase():
    path = r"D:\New folder (4)\computer engineering 195339\GRADII\line 54"
    detected_faces = []
    for i in range(1, 2):
        img = cv2.cvtColor(cv2.imread(os.path.join(path,'Frame' + str(i) + '.jpg')), cv2.COLOR_BGR2RGB)
        detector = MTCNN(min_face_size=15,steps_threshold=[0.05,0.25,0.845])
        faces = detector.detect_faces(img)
        draw_facebox(img, faces)

        detected_faces.append([face["box"] for face in faces])
    return detected_faces
#------------------------------------------------------------------------------------------------------------#
def draw_facebox(img, result_list):
    path = r"D:\New folder (4)\computer engineering 195339\GRADII\line 54"
    # load the image
    number_of_faces = 0
    data = img
    # plot the image
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for result in result_list:
        # get coordinates
      
        x, y, width, height = result['box']
        # create the shape
        rect = plt.Rectangle((x, y), width, height, fill=False, color='green')
        # draw the box
        ax.add_patch(rect)
        # crop the face from the image
        face = data[y:y+height, x:x+width]
        # save the face to a separate folder
        filename = os.path.join(path, f"face_{number_of_faces}.jpg")
        cv2.imwrite(filename, face, [cv2.IMWRITE_JPEG_QUALITY, 100])
        number_of_faces += 1

   
    # save the image with face boxes drawn on it
    cv2.imwrite(os.path.join(path, 'Frame with faces specified.jpg'), data, [cv2.IMWRITE_JPEG_QUALITY, 100])


#------------------------------------------------------------------------------------------------------------#


import cv2
import os




def recognition_phase(faces_locations) :
    
    students_data_dir = r'D:\Downloads Folder\test 72' #Should be the picture of real students folder not the captured frames
    student_data_generator = ImageDataGenerator(rescale=1./255)
    val_batch_size = 10
    val_img_size = (125, 125)

    val_generator = student_data_generator.flow_from_directory(
    directory=students_data_dir,
    target_size=val_img_size,
    batch_size=val_batch_size,
    class_mode='categorical',
    shuffle=False)

    identity_labels_dictionary = val_generator.class_indices


    model = load_model(r'C:\Users\Atef\Desktop\computer engineering 195339\GRADII\recognitionFinal.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    preprocessed_faces=[]
    attended_students=[]
    for i in range (0,31):
        
        for top, right, bottom, left in faces_locations :
            face = faces_locations[top, right, bottom, left]
            resized_face = cv2.resize(face, (125, 125))
            normalized_face = resized_face / 255.0
            finalface=np.expand_dims(normalized_face, axis=0)
            preprocessed_faces.append(normalized_face)
            probs = model.predict(finalface)
            pred_class_idx = np.argmax(probs)

            for key, value in identity_labels_dictionary.items():
            # Check if the value matches the given value
                if value == pred_class_idx and value not in attended_students:
                    # Print the key corresponding to the value
                        attended_students.append(key)
#------------------------------------------------------------------------------------------------------------#

#Main Code
#webcam_connection ()
faces_locations = detection_phase()
#recognition_phase(faces_locations)