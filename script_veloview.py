import time
import os
import pyautogui
from multiprocessing.pool import ThreadPool

def applogic() -> None:
    os.system("/home/ubuntu/Downloads/VeloView-4.1.3-Linux-64bit/bin/VeloView --script=/home/ubuntu/Downloads/lidar/new_applogic.py ")
    
def press_key() -> None:
    time.sleep(3)
    pyautogui.press('f8')
    time.sleep(2)
    pyautogui.typewrite('capture()')
    pyautogui.press('enter')

if __name__ == '__main__':
    thread_pool = ThreadPool(processes=4)
    thread_applogic = thread_pool.apply_async(applogic, args=())
    thread_keyPress = thread_pool.apply_async(press_key, args=())
    
    thread_applogic.get()
    thread_keyPress.get()
    