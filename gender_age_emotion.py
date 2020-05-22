
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import img_to_array
face_classifier = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')




class_labels = ['Angry','Happy','Neutral','Sad','Surprise']


classifier= load_model('./weights/Emotion_little_vgg.h5')


# In[6]:


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


# In[10]:


faceProto="./models/opencv_face_detector.pbtxt"
faceModel="./models/opencv_face_detector_uint8.pb"
ageProto="./models/age_deploy.prototxt"
ageModel="./models/age_net.caffemodel"
genderProto="./models/gender_deploy.prototxt"
genderModel="./models/gender_net.caffemodel"


# In[11]:


MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']


# In[12]:


faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


# In[13]:


padding=20


# In[ ]:





# In[ ]:





# ## Running code for the real time gender age and emotion detection

# In[14]:


video=cv2.VideoCapture(0)
padding=20
while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break        
    resultImg,faceBoxes=highlightFace(faceNet,frame)   #resultImg :- frame pe rectangle
                                                       #faceboxex :- (x,y,w,h) pairs
    if not faceBoxes:
        print("No face detected")
    
    for faceBox in faceBoxes:
        
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
        
       
    
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)        
        
        
        #emotions
        gray_face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(gray_face,(48,48),interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            preds = classifier.predict(roi)[0]
            Emotion=class_labels[preds.argmax()]
            print(f'Emotions : {Emotion}')
        
        
        #detect gender
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')
        
        #detect age
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        
        
        #showing results
        cv2.putText(resultImg, f'{gender}, {age},{Emotion}', (faceBox[0]-50, faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age, gender and emotion ", resultImg)
    
    
    # to quit
    key_pressed = cv2.waitKey(1) & 0xff
    if key_pressed==ord('q'):
        break
    
    
# to destroy all windows        
video.release()
cv2.destroyAllWindows()

