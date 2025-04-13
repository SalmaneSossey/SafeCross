# SafeCross
A project that helps blind people cross any kind of streets. 

# You need following requirements to run the project : 

_*Hardware*_:
- Raspberry Pi 4
- SD Card for OS
- Camera USB
- 2 motor vibrators (Mini Vibration Motor Module (DC Motor with PCB Driver Board))
- Power Bank

_*Software*_:
- Python 3.11
- OpenCV for python
- Numpy
- .caffemodel and .prototxt

You need to download from this git the .caffemodel and .prototxt

If you want to run the project on windows, then download "python windows", it will not send vibration, but instead, it will show a live stream with objects framed with their classification and percentage, and marked as green if they are in movement, or as blue if they are static.
It will not require motor vibrators because the goal of this folder is to study the software and the input, you can choose either camera usb or laptop camera, or even a pre-recorded video, you can actually take the video's path and put it in the code.
Then you execute `pip install numpy opencv-python` to install necessary libraries

Concerning raspberry pi 4, you will need to install an OS on SD Card, and then set your username, hostname and password, and also the wifi to connect.
So you connect to it via wifi and using cmd or putty, then u can control it and access it, and control it with commands just like you do in Ubuntu Terminal
Then you transfer the files into raspberry.
Next step is to create virtual environment, but before all that, make sure you have updated the OS, then you activate the env, and u execute the `main.py` file, make sure to connect all components to raspberry and power bank as well.
There are many tutorials that show how to connect to it with wifi to control it, and to deploy files, etc..
Then you can set the OS to execute the `main.py` once it is booted.

Optimizations may be made in future inchaallah
