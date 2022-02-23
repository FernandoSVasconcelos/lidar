#!/usr/bin/python

import csv
import glob
import numpy as np
import os
import pandas as pd
import socket
import struct
import sys
import time
import traceback
from socket import socket, AF_INET, SOCK_DGRAM

from datetime import datetime, timedelta
from multiprocessing import Process, Queue, Pool
from constantes import (
    DISTANCE_RESOLUTION, EXPECTED_PACKET_TIME, EXPECTED_SCAN_DURATION, 
    HOST, LASER_ANGLES, NUM_LASERS, PORT, 
    ROTATION_MAX_UNITS, ROTATION_RESOLUTION
)

class Lidar:
    def __calc(self, dis, azimuth, laser_id, timestamp, reflectivity):
        R = dis * 360 #0.0002
        laser_angles = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
        omega = laser_angles[laser_id] * np.pi / 180.0
        alpha = azimuth / 100.0 * np.pi / 180.0
        X = R * np.cos(omega) * np.sin(alpha)
        Y = R * np.cos(omega) * np.cos(alpha)
        Z = R * np.sin(omega)
        return [X, Y, Z, azimuth, dis, laser_id, reflectivity]

    def __capture(self, port):
        try:
            socketfd = socket(AF_INET, SOCK_DGRAM) 
            socketfd.bind(('', port))
            socketfd.settimeout(1)
	
            try:
                data = []
                for i in range(200):
                    data.append(socketfd.recv(2000)) # Cada captura realiza aproximadamente 10 graus  -> volta completa = 40 capturas
                    if len(data[i]) > 0:
                        assert len(data[i]) == 1206, len(data[i])
                        
                socketfd.close()
                return data
            except Exception as exception:
                socketfd.close()
                print(f"[__capture] Exception - {getattr(exception, 'message', str(exception))}")
                return None
            
        except Exception as exception:
            print(f"[__capture] Exception - {getattr(exception, 'message', str(exception))}")
            return None

    def __unpack(self, binary):
        chunks = []
        
        for bits in binary:
            points = []
            scan_index = 0
            prev_azimuth = None
            n = len(bits)
            for offset in range(0, n, 1223):
                data = bits[offset: offset + 1223]
                timestamp, factory = struct.unpack_from("<IH", data, offset=1200)
                assert factory == 0x2237, hex(factory)  # 0x22=VLP-16, 0x37=Strongest Return
                seq_index = 0
                for offset in range(0, 1200, 100):
                    flag, azimuth = struct.unpack_from("<HH", data, offset)
                    assert flag == 0xEEFF, hex(flag)
                    for step in range(2):
                        seq_index += 1
                        azimuth += step
                        azimuth %= 36000               
                        prev_azimuth = azimuth
                        # H-distance (2mm step), B-reflectivity (0
                        arr = struct.unpack_from('<' + "HB" * 16, data, offset + 4 + step * 48)
                        reflectivity = (struct.unpack_from('<HB', data, offset)[1])
                        
                        for i in range(16):
                            time_offset = (55.296 * seq_index + 2.304 * i) / 1000000.0
                            if arr[i * 2] != 0:
                                calc_return = self.__calc(arr[i * 2], azimuth, i, timestamp + time_offset, reflectivity)
                                points.append(calc_return)
            chunks.append(points)

        flatten = [item for sublist in chunks for item in sublist]
        return flatten

    def data(self):
        try:
            binary = self.__capture(2368)
            if binary == None:
                print(f"[data] - Falha em obter os dados binários do Lidar.")
                return None

            result = self.__unpack(binary)
            df = pd.DataFrame(result[1:], columns=['X', 'Y', 'Z', 'azimuth', 'dis', 'laser_id', 'reflectivity'])
            df.to_csv('teste.csv')
            return df
        except Exception as exception:
            print(f"[data] - {exception}")
            return None

if __name__ == '__main__':
    newLidar = Lidar()
    newLidar.data()
