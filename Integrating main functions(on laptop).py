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
import glob

After15min=False

def webcam_connection (path):
    global After15min
    webCam = cv2.VideoCapture(0)
    currentframe = 1
    if After15min == False :
        path = r"D:\New folder (4)\computer engineering 195339\Level 5\GRADII\Test unit\Test Unit01\Deep\Cc"
        while (currentframe<15):
            #read from webcam
            success, frame = webCam.read()
            # Save Frame by Frame into disk using imwrite method
            time.sleep(1)
            cv2.imwrite(os.path.join(path,'Frame' + str(currentframe) + '.jpg'), frame) #save each captuerd frame in frames folder 
            currentframe += 1  # Inc the number of saved farmes to keep track of the number of frames
            time.sleep(59)  # Wait 59 second between 2 frames

    elif After15min == False :
         path = r"D:\New folder (4)\computer engineering 195339\Level 5\GRADII\Test unit\Test Unit01\Deep\Cc2"
         while (currentframe<6):
            #read from webcam
            success, frame = webCam.read()
            # Save Frame by Frame into disk using imwrite method
            time.sleep(1)
            cv2.imwrite(os.path.join(path,'Frame' + str(currentframe) + '.jpg'), frame) #save each captuerd frame in frames folder 
            currentframe += 1  # Inc the number of saved farmes to keep track of the number of frames
            time.sleep(599)  # Wait 59 second between 2 frames



def detection_phase():
    global After15min
    if After15min == False :
         path = r"D:\New folder (4)\computer engineering 195339\Level 5\GRADII\Test unit\Test Unit01\Deep\Cc"
    elif After15min == True :
        path = r"D:\New folder (4)\computer engineering 195339\Level 5\GRADII\Test unit\Test Unit01\Deep\Cc2"
    print (path)
    files = glob.glob(os.path.join(path, "*"))
    num_of_pictures = len(files)
    print (num_of_pictures)
    detected_faces = [[] for _ in range(num_of_pictures)]
    for i in range(1, (num_of_pictures+1 )):
        img = cv2.cvtColor(cv2.imread(path +"\Frame"+ str(i) +".jpg"), cv2.COLOR_BGR2RGB)

        detector = MTCNN(min_face_size=26)#min_face_size=15,steps_threshold=[0.01,0.33,0.8]
        faces = detector.detect_faces(img)
        draw_facebox(img, faces)
        for result in faces:
            detected_faces[i-1].append(result["box"])
    return detected_faces

#------------------------------------------------------------------------------------------------------------#total_faces = 0
total_faces = 0

def draw_facebox(img , result_list):
    global total_faces
    global After15min

    if After15min == False :
         path = r"D:\New folder (4)\computer engineering 195339\Level 5\GRADII\Test unit\Test Unit01\Deep\Cc"
    elif After15min == True :
        path = r"D:\New folder (4)\computer engineering 195339\Level 5\GRADII\Test unit\Test Unit01\Deep\Cc2"
 
    faces_path = os.path.join(path, "faces to recognize")

    # create the 'faces to recognize' folder if it doesn't exist
    if not os.path.exists(faces_path):
        os.makedirs(faces_path)

    # load the image
    number_of_faces = 0
    data = img 

    # clear the axis
    ax = plt.gca()
    ax.clear()

    # plot the image
    plt.imshow(data)
    
    # get the context for drawing boxes+
    ax = plt.gca()

    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = plt.Rectangle((x+2, y+2), width, height, fill=False, color='green')
        # draw the box
        ax.add_patch(rect)
        face = data[y:y+height+2, x:x+width+2]
        # convert face to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # save the face to the 'faces to recognize' folder
        filename = os.path.join(faces_path, f"face_detected{total_faces}.jpg")
        cv2.imwrite(filename, face, [cv2.IMWRITE_JPEG_QUALITY, 200])
        total_faces += 1
        number_of_faces += 1
        
    plt.savefig(os.path.join(path, 'Frame' + f"{number_of_faces}"+f"face_detected{total_faces}" + 'with faces specified' + '.jpg'), dpi=300)


def recognition_phase ():
    global After15min
    temp= []
    if After15min == False :
        path = r"D:\New folder (4)\computer engineering 195339\Level 5\GRADII\Test unit\Test Unit01\Deep\Cc"
    elif After15min == True :
        path = r"D:\New folder (4)\computer engineering 195339\Level 5\GRADII\Test unit\Test Unit01\Deep\Cc2"

    # Load the saved model
    model = load_model(r'C:\Users\Atef\Desktop\recognition final 7-6-2023 94%(Utopia Last day)3.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #registered course students directory
    students_data_dir = r'D:\New folder (4)\computer engineering 195339\Level 5\GRADII\Transfer Learning data\utopia\uuu' #Should be the picture of real students folder not the captured frames
    student_data_generator = ImageDataGenerator( rescale=1./255,
)

    val_batch_size = 1
    val_img_size = (125, 125)

    val_generator = student_data_generator.flow_from_directory(
    directory=students_data_dir,
    target_size=val_img_size,
    batch_size=val_batch_size,
    class_mode='categorical',
    shuffle=False)

    identity_labels_dictionary = val_generator.class_indices
    un_recognized=0
    attended_students=[]
    
    faces_path = os.path.join(path, "faces to recognize")
    files = glob.glob(os.path.join(faces_path, "*"))
    num_of_pictures = len(files)
    print(num_of_pictures)
  
    for i in range(0, (num_of_pictures )):

        img_path = os.path.join(faces_path, f"face_detected{i}.jpg")
        print (img_path)
        # Load the image and resize it to match the input size of the model
        img = image.load_img(img_path, target_size=(125, 125))

        # Convert the image to a numpy array and normalize the pixel values
        img_array = image.img_to_array(img) / 255.

        # Add a new dimension to the array to match the input shape of the model
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction using the model and obtain the predicted class probabilities
        probs = model.predict(img_array)
        print (probs)
        # Obtain the predicted class index and identity label
        pred_class_idx = np.argmax(probs)
        
        
        if probs[0][pred_class_idx] > 0.8 :

            identity_labels_dictionary = val_generator.class_indices

            # Iterate over the key-value pairs in the dictionary
            for key, value in identity_labels_dictionary.items():
                # Check if the value matches the given value
                if value == pred_class_idx:
                    # Print the key corresponding to the value
                    print('the identity is:', key)
                    if key not  in attended_students and After15min == False :
                        attended_students.append(key)
                        print(key)
                    elif  After15min == True :
                       temp.append(key) 
                       print(key)
                       if temp.count(key) > 1 and key not  in attended_students : 
                            attended_students.append(key) 

        else :
            un_recognized=un_recognized+1
        print(f'unrecognized faces = {un_recognized}')
    return attended_students


#webcam_connection ()  
faces_locations = detection_phase()
att1 =recognition_phase()
print(att1)
print(len((att1)))

After15min = True 
######################################################
total_faces = 0
#webcam_connection ()  
faces_locations = detection_phase()
att2 =recognition_phase()
print(att2)
print(len((att2)))

######################################################
final_attendance = list(set(att1).intersection(att2))
print(final_attendance)