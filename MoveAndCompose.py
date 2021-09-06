# -*- coding: utf-8 -*-
# @Time : 2021/9/6 17:15
# @Author : Mingjun Xiang
# @Site : 
# @File : MoveAndCompose.py
# @Software: PyCharm 
# @Illustration: Combine high exposure and low exposure sky images
#                which will be moved to target directory.
#                The same exposure-time images will be labeled as wrong.
#                The illustration of hdr will be updated when i'm free.
#                The acquisition of hdr parameters is hard to explain,
#                maybe one day i will rewrite, now please use the current data.


import os
from hdr import *
from PIL import Image
import numpy as np
import multiprocessing


def fun(files, folder, path, path1, path2, path3):
    img1, img2 = os.path.join(path1, folder, files[0]), os.path.join(path1, folder, files[1])
    name = files[2]
    images = []
    if os.path.exists(img1) and os.path.exists(img2):
        try:
            images.append(np.array(Image.open(img1)))
            images.append(np.array(Image.open(img2)))
            mean1, mean2 = np.mean(images[0]), np.mean(images[1])
            if isinstance(mean1, np.float64) and isinstance(mean2, np.float64) and abs(mean1 - mean2) > 10:
                # decide exposure time (important)
                log_exposure_times = np.array([log(1), log(2)])
                o = computeHDR(images, log_exposure_times, state='load', path=path, smoothing_lambda=100.,
                               gamma=0.6,
                               eps=40)
                o = Image.fromarray(o)
                o.save(os.path.join(path2, name + '.jpg'))
                # import matplotlib.pyplot as plt
                # plt.imshow(np.array(o))
                # plt.show()
                print(name[:12])
            else:
                o = Image.fromarray(images[0])
                o.save(os.path.join(path3, name + '.jpg'))
                o = Image.fromarray(images[1])
                o.save(os.path.join(path3, name + '_low.jpg'))
                print(name[:12])
            return False
        except Exception:
            print(1)
            return True


if __name__ == '__main__':
    # origin path
    path1 = r'./output/origin'
    # path for saving valid images
    path2 = r'./output/valid'
    # path for saving abnormal images
    path3 = r'./output/wrong_image'
    # hdr parameters
    path = r'./HdrParameters'

    if not os.path.exists(path2):
        os.makedirs(path2)
    if not os.path.exists(path3):
        os.makedirs(path3)

    folders = os.listdir(path1)
    folders = [folder for folder in folders if not folder.find('.') != -1]
    a = []
    for folder in folders:
        list_files = os.listdir(os.path.join(path1, folder))
        list_files2 = [x[0:14] for x in list_files]
        list_files2 = list(set(list_files2))
        list_files3 = [[x + '_12.jpg', x + '_11.jpg', x] for x in list_files2]
        pool = multiprocessing.Pool()
        for files in list_files3:
            res = pool.apply_async(fun, args=(files, folder, path, path1, path2, path3))
            # res = fun(files, folder, path, path1, path2, path3)
        pool.close()
        pool.join()
        if res:
            a.append(files[2])

    print(a)
