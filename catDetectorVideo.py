# OpenCV program to detect cat face in real time 
# import libraries of python OpenCV 
# where its functionality resides 
import cv2

# load the required trained XML classifiers 
# https://github.com/Itseez/opencv/blob/master/ 
# data/haarcascades/haarcascade_frontalcatface.xml 
# Trained XML classifiers describes some features of some 
# object we want to detect a cascade function is trained 
# from a lot of positive(faces) and negative(non-faces) 
# images. 
detector = cv2.CascadeClassifier('./classifier/haarcascade_frontalcatface_extended.xml')

# capture frames from a camera 
cap = cv2.VideoCapture(0) 

# loop runs if capturing has been initialized.
img_counter = 0
while 1: 

    # reads frames from a camera 
    ret, img = cap.read() 

    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Flip image verticaly
    flip = cv2.flip(gray, 0);

    # Detects cats of different sizes in the input image
    rects = detector.detectMultiScale(flip, scaleFactor=1.3, minNeighbors=5, minSize=(75, 75))

    for (i, (x, y, w, h)) in enumerate(rects):
        # To draw a rectangle in a face 
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.putText(img, "Chat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        
        # Save image with the cat
        img_name = "pictures/Chat20201221_{}.png".format(img_counter)
        cv2.imwrite(img_name, flip)
        img_counter += 1

        
        



    # Display an image in a window 
    cv2.imshow('img',flip) 

    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

# Close the window 
cap.release() 

# De-allocate any associated memory usage 
cv2.destroyAllWindows() 

