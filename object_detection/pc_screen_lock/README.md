# PC Screen Lock using Face Recognition
This is the first fun implementation of Practical ML Implementations project. 
This ML implementation aims to show how cool stuff could be built using simple existing open-source libraries.
   
## Introduction:
This implementation locks down your PC screen when you go away from your own PC and if someone attemps to look at your PC.

## Requirements:
- [Face Recognition package](https://github.com/ageitgey/face_recognition)
    - Python 3.3+ or Python 2.7
    - macOS or Linux (Windows not officially supported, but might work)
- Tested on only MacbookPro, macOS Catalina

## Install Face Recognition package
- First, make sure you have dlib already installed with Python bindings: [install dlib](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)
- git clone https://github.com/ageitgey/face_recognition.git
- pip3 install face_recognition
- pip install git+https://github.com/ageitgey/face_recognition_models

Check [Face Recognition repository]((https://github.com/ageitgey/face_recognition)) for other options (including Docker, pre-configured VM, Nvidia Jetson Nano, Raspberry Pi 2+, etc) to install face_recognition.

## How to run:
- Store your own face image at `./known-faces/` folder, name the images with your name (e.g., `alisher.jpg`)
- Run the script with [caffeinate](http://lightheadsw.com/caffeine/): `caffeinate python lock_ps_screen.py`
- If any unknown people appear in front of your PC, your PC will go to sleep :) Takes the snapshot of "unknown" person and stores the corresponding face under `./unkown-faces/`  folder with the timestamp.
- When you are back to your PC, program will continue running (no need to restart the python script)
- You can check who attempted to look at your PC Screen when were not there :)
- Have Fun!