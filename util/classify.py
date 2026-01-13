import os
import numpy as np

import argparse
import multiprocessing
from multiprocessing import Pool
import shutil
if __name__ == '__main__':
    path_F ='/home/bj/pix2pix/test_result/dists/fakes'
    path_R = '/home/bj/pix2pix/test_result/dists/reals'
    path = '/home/bj/pix2pix/results/dists/test_latest/images'
    os.makedirs(path_F, exist_ok=True)
    os.makedirs(path_R, exist_ok=True)
    i = 1 
    j = 1
    for filename in os.listdir(path):
        source_file_path = os.path.join(path,filename)
        if "fake" in filename and "B" in filename:
            new_filename =filename.replace('fake','')
            destination_file_path = os.path.join(path_F, new_filename)
            shutil.copy2(source_file_path, destination_file_path)
        if "real" in filename and "B" in filename:
            new_filename = filename.replace('real','')
            destination_file_path = os.path.join(path_R, new_filename)
            shutil.copy2(source_file_path, destination_file_path)