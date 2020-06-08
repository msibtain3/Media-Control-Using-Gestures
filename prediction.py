import vlc
import cv2
from tensorflow import keras
import tensorflow as tf
from keras.models import load_model
import operator
import numpy
import time
import RPi.GPIO as GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11,GPIO.OUT)
GPIO.setup(13, GPIO.OUT)
from time import sleep

Instance = vlc.Instance()
player = Instance.media_player_new()
Media = Instance.media_new("/home/pi/Downloads/video.mp4")
player.set_media(Media)
player.play()

classifier = tf.keras.models.load_model('/home/pi/Downloads/model.h5')
cap = cv2.VideoCapture(0)
# Category dictionary
class_labels = ["ONE",'TWO','THREE','FOUR','FIVE']
while True:
    ret, frame = cap.read()
        # Simulating mirror image
    frame = cv2.flip(frame, 1)

        #getting roi of the hand part
    x1 = 400
    y1 = 50
    x2 = 600
    y2 = 300
    cv2.rectangle(frame , (x1-2,y1-2),(x2+2,y2+2),(0,255,0),2)

    # extracting the roi and converting it to gray
    roi = frame[y1:y2 , x1:x2]
    roi = cv2.resize(roi,(150,150))
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

    # applying a threshold to the region of interest
    ret,test_image = cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('test',test_image)
    result = classifier.predict(test_image.reshape(1, 150, 150, 1))
    prediction = {'ONE': result[0][0], 
                  'TWO': result[0][1], 
                  'THREE': result[0][2],
                  'FOUR': result[0][3],
                  'FIVE': result[0][4]}
        # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
        # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (100, 450), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 4)
    print(prediction[0][0], )
    if prediction[0][0] == "THREE":
        player.play()
        #player.audio_set_volume(100)
    elif prediction[0][0] == "FOUR":
        player.stop()
        
        GPIO.output(11, True)
        GPIO.output(13, True)
        time.sleep(3)
        GPIO.output(11, False)
        GPIO.output(13,False)
        time.sleep(2)
        #player.audio_set_volume(0)
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(1)
    if interrupt & 0xFF == 27: # esc key
        break
cap.release()
cv2.destroyAllWindows()

