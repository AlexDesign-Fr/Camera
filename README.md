# Camera
Intelligent cat feeder with cats recognition.

## Goal
This programm is a cat's detector to feed them when they pass in front of a Raspberry Pi camera.
It will distinguish Cat1 from Cat2 and give the appropriate food according different cat. 

# Installation
Compiling OpenCV for Raspbery Pi 4 is not so easy, please refer to [my (french) web site](https://www.alex-design.fr/Projets-R-A/Nourrisseur-intelligent-pour-chats/Installation-de-OpenCV-sur-une-Raspberry-Pi-4) for that.

# Programs description
## catPicturesCapture.py
OpenCV program to detect cat face in real time.
When a cat is detected, we save it's face in a file. The file is named with the current date to be unique.

These images will be used by machine-learning to distinguish Cat1 from Cat2