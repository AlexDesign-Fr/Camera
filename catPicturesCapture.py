# OpenCV program to detect cat face in real time 
# import libraries of python OpenCV 
# where its functionality resides 
import cv2
from datetime import datetime

# load the required trained XML classifiers 
# https://github.com/Itseez/opencv/blob/master/ 
# data/haarcascades/haarcascade_frontalcatface.xml 
# Trained XML classifiers describes some features of some 
# object we want to detect a cascade function is trained 
# from a lot of positive(faces) and negative(non-faces) 
# images. 
detector = cv2.CascadeClassifier('./classifier/haarcascade_frontalcatface_extended.xml')

# capture frames from a camera 
videoCapture = cv2.VideoCapture(0)

# Check if video can be opened
if not videoCapture.isOpened():
    raise Exception("Could not open video device")

# Set properties. Each returns === True on success (i.e. correct resolution)
videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

# loop runs if capturing has been initialized.
while 1:

    # reads frames from the Raspi camera
    ret, img = videoCapture.read()

    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Flip image verticaly
    # 0 : vertical flipping
    # 1 : horizontal flipping
    # -1 : Both
    flip = cv2.flip(gray, 0);

    # Detects cats of different sizes in the input image
    # minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    # minSize – Minimum possible object size. Objects smaller than that are ignored
    # maxSize – Maximum possible object size. Objects bigger than this are ignored
    rects = detector.detectMultiScale(flip, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (i, (x, y, w, h)) in enumerate(rects):
        date = datetime.now()

        # Save image with the cat
        img_name = "pictures/Chat_{}.png".format(date)
        cv2.imwrite(img_name, flip)


    # Display an image in a window
    cv2.imshow('video live',flip)

    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

# Close the window 
videoCapture.release()

# De-allocate any associated memory usage 
cv2.destroyAllWindows() 

