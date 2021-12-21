import time
import os
import keyboard
import _thread

def applogic():
    os.system("/home/ubuntu/Downloads/VeloView-4.1.3-Linux-64bit/bin/VeloView --script=/home/ubuntu/Downloads/new_applogic.py ")
    

def press_key():
    keyboard.send('f8')
    keyboard.write('teste')

if __name__ == '__main__':
    _thread.start_new_thread(applogic, ())
    time.sleep(2)
    _thread.start_new_thread(press_key, ())
    