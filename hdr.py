# -*- coding: utf-8 -*-
# @Time : 2021/9/6 17:33
# @Author : Mingjun Xiang
# @Site :
# @File : hdr.py
# @Software: PyCharm
# @Illustration: hdr composing with stacked images


import numpy as np
import cv2
import random
from math import log, sqrt
import os
import time
from scipy.sparse import csr_matrix
# import line_profile

"""
Based on vivianhylee's hdr code
https://github.com/vivianhylee/high-dynamic-range-image/blob/master/hdr.py
I have change some codes to speed up the program.
My PC has turned 500s to 10s to process the

How to solve:
Step1: find the minimum cost F.
    N P
F = ΣΣ{W(Z_ij)[g(Z_ij)-ln(E_i)-ln(t_j)]}^2+λW(Z_ij)g''(z)^2
    i j 
    where g():the function let g(Z)=lnEt
        Z_ij: value of ith pixel in jth image
        E_i: value of intensity at ith pixel
        t_j: value of exposure time at jth image
        λ: smoothing coefficient
        g''(z): g(z-1)-2g(z)+g(z+1)
        N: total pixels in one image
        P: number of images
        W(Z_ij): coefficient to decrease the impact of inaccurate value

Step2: considering that 256 values of g(z) and intensity_range 
    number(E_i, just define as 256) variables(total 512).
    Quadratic cost function differential changed the problem into Ax=b.
    But this code is not completely differential every element,
    refer to function computeResponseCurve.
    Thus we get the value of g(Z_ij), we need to use this to recalculate
    ln(E_i)

Step3: recalculate ln(E_i)
                P
                ΣW(Z_ij)[g(Z_ij)-ln(t_j)]
                j
    ln(E_i) = ------------------------------
                        P
                        ΣW(Z_ij)
                        j
    
Step4: gamma correction and the mean to be same
#


"""


def cmask(index, radius, array):
    a, b = index
    is_rgb = len(array.shape)

    if is_rgb == 3:
        ash = array.shape
        nx = ash[0]
        ny = ash[1]

    else:
        nx, ny = array.shape

    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    mask = x * x + y * y <= radius * radius
    return mask


def linearWeight(z_min=0, z_max=255, mid=150):
    """ Linear weighting function based on pixel intensity that reduces the
    weight of pixel values that are near saturation.

    Parameters
    ----------
    z_min, z_max : np.uint8
        A pixel intensity value from 0 to 255

    Returns
    -------
    weight : <numpy.ndarray>, dtype=np.unit8, shape=(z_max-z_min+1)
        The weight corresponding to the input pixel intensity
    W = [0 1 2 ... 126 127 127 126 ... 2 1 0]
    """
    weight1 = np.arange(z_min, z_max+1)
    weight2 = weight1[::-1]
    return np.minimum(weight1, weight2)


def sampleIntensities(images):
    """Randomly sample pixel intensities from the exposure stack.

    The mid col represent the value of randomly chosen point, and others
    are corresponding value on other images

    Parameters
    ----------
    images : <numpy.ndarray>, shape=(num_images, num_length, num_height)
        Images containing a stack of single-channel (i.e., grayscale)
        layers of an HDR exposure stack

    Returns
    -------
    intensity_values : <numpy.ndarray>, dtype=np.uint8,
                shape = (num_intensities, num_images)
        An array containing a uniformly sampled intensity value from each
        exposure layer (shape = num_intensities x num_images)
    IV = [[    0   0   0   0   1]
            [  0   1   1   2   5]
              ..................
            [239 241 254 255 255]
            [240 241 255 255 255]]
    """
    z_min, z_max = 0, 255
    num_intensities = z_max - z_min + 1

    num_images = images.shape[0]

    intensity_values = np.arange(z_min, z_max+1).reshape(num_intensities, 1) \
                       * np.ones((num_intensities, num_images), dtype=np.uint8)

    # Find the middle image to use as the source for pixel intensity locations
    mid_img = images[num_images // 2]

    # 采用稀疏矩阵的性质csr_matrix((num, (row, col)), shape=(range, num))实现提速 不用np.where
    # np.unravel_index将一维的索引恢复为多维数组的索引
    # 这一段耗时最长 约0.26s
    cols = np.arange(mid_img.size)
    sparse = csr_matrix((cols, (mid_img.flatten(), cols)), shape=(num_intensities, mid_img.size))
    locations = [np.unravel_index(row.data, mid_img.shape) for row in sparse]

    for i in range(z_min, z_max + 1):
        rows, cols = locations[i]
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            intensity_values[i, :] = images[:, rows[idx], cols[idx]]

    return intensity_values


def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weight):
    """Find the camera response curve for a single color channel

    Parameters
    ----------
    intensity_samples : <numpy.ndarray>
        Stack of single channel input values (num_samples x num_images)

    log_exposures : <numpy.ndarray>
        Log exposure times (size == num_images)

    smoothing_lambda : float
        A constant value used to correct for scale differences between
        data and smoothing terms in the constraint matrix -- source
        paper suggests a value of 100.

    weight : <numpy.ndarray>, dtype=np.unit8, shape=(z_max-z_min+1)
        The weight corresponding to the input pixel intensity

    Returns
    -------
    numpy.ndarray, dtype=np.float64
        Return a vector g(z) where the element at index i is the log exposure
        of a pixel with intensity value z = i (e.g., g[0] is the log exposure
        of z=0, g[1] is the log exposure of z=1, etc.)
    """
    z_min, z_max = 0, 255
    intensity_range = 255  # difference between min and max possible pixel value for uint8
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)

    # NxP + [(Zmax-1) - (Zmin + 1)] + 1 constraints; N + 256 columns
    # 变量个数为g(Z_ij)的个数 + 255份光强(intensity_range)的个数
    # mat_A = np.zeros((num_images * num_samples + intensity_range, num_samples + intensity_range + 1), dtype=np.float64)
    mat_A = np.zeros((num_images * num_samples + z_max - z_min, num_samples + intensity_range + 1), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    # 1. Add data-fitting constraints:
    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            z_ij = intensity_samples[i, j]
            w_ij = weight[z_ij]
            mat_A[k, z_ij] = w_ij
            mat_A[k, (intensity_range + 1) + i] = -w_ij
            mat_b[k, 0] = w_ij * log_exposures[j]
            k += 1

    # 2. Add smoothing constraints:
    for z_k in range(z_min + 1, z_max):
        w_k = weight[z_k]
        mat_A[k, z_k - 1] = w_k * smoothing_lambda
        mat_A[k, z_k    ] = -2 * w_k * smoothing_lambda
        mat_A[k, z_k + 1] = w_k * smoothing_lambda
        k += 1

    # 3. Add color curve centering constraint:
    mat_A[k, (z_max - z_min) // 2] = 1

    # cal pin cost several time
    inv_A = np.linalg.pinv(mat_A)
    x = np.dot(inv_A, mat_b)
    g = x[z_min: z_max + 1]

    return g[:, 0]


def computeRadianceMap(images, log_exposure_times, response_curve, eps):
    """Calculate a radiance map for each pixel from the response curve.

    Parameters
    ----------
    images : list
        Collection containing a single color layer (i.e., grayscale)
        from each image in the exposure stack. (size == num_images)

    log_exposure_times : numpy.ndarray
        Array containing the log exposure times for each image in the
        exposure stack (size == num_images)

    response_curve : numpy.ndarray
        Least-squares fitted log exposure of each pixel value z

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        The image radiance map (in log space)
    list
        The place I will use 255 and 0 as output
    """
    s = time.time()
    images = np.array(images)
    dim = images.shape

    T = np.swapaxes(log_exposure_times * np.ones(dim[::-1]), 0, 2)
    G = response_curve[images]
    # W = weight[images]
    W = np.minimum(images, 255-images) + 1e-6

    # Important, this will eliminate the influence of the gray phenomenon at whitest part
    max_exposure = np.argmax(log_exposure_times)
    min_exposure = np.argmin(log_exposure_times)
    max_place = images[min_exposure] >= 255 - eps
    min_place = images[max_exposure] <= eps

    img_rad_map = np.divide(np.sum(W * (G - T), axis=0), np.sum(W, axis=0))

    return img_rad_map, [max_place, min_place]


def globalToneMapping(image, gamma):
    """Global tone mapping using gamma correction
    ----------
    images : <numpy.ndarray>
        Image needed to be corrected
    gamma : floating number
        The number for gamma correction. Higher value for brighter result; lower for darker
    Returns
    -------
    numpy.ndarray
        The resulting image after gamma correction
    """
    image = np.abs(image)
    image_corrected = cv2.pow(image/255., 1.0/gamma)
    return image_corrected


def intensityAdjustment(image, template):
    """Tune image intensity based on template
        ----------
        images : <numpy.ndarray>
            image needed to be adjusted
        template : <numpy.ndarray>
            Typically we use the middle image from image stack. We want to match the image
            intensity for each channel to template's
        Returns
        -------
        numpy.ndarray
            The resulting image after intensity adjustment
        """
    m, n, channel = image.shape
    output = np.zeros((m, n, channel))
    for ch in range(channel):
        image_avg, template_avg = np.mean(image[:, :, ch]), np.mean(template[:, :, ch])
        output[..., ch] = image[..., ch] * (template_avg / image_avg)

    return output


def computeHDR(images, log_exposure_times, state='save', path=None, smoothing_lambda=100., gamma=0.6, eps=40):
    """Computational pipeline to produce the HDR images
    ----------
    images : list<numpy.ndarray>
        A list containing an exposure stack of images
    log_exposure_times : numpy.ndarray
        The log exposure times for each image in the exposure stack
    smoothing_lambda : np.int (Optional)
        A constant value to correct for scale differences between
        data and smoothing terms in the constraint matrix -- source
        paper suggests a value of 100.
    Returns
    -------
    numpy.ndarray
        The resulting HDR with intensities scaled to fit uint8 range
    """

    height, width, num_channels = images[0].shape
    hdr_image = np.zeros_like(images[0])
    Weight = linearWeight()

    places = []
    layer_stacks = []
    for channel in range(num_channels):

        # Collect the current layer of each input image from the exposure stack
        layer_stack = [img[:, :, channel] for img in images]
        layer_stacks.append(layer_stack)
        layer_stack = np.array(layer_stack)

        # Sample image intensities
        if state == 'load':
            intensity_samples = np.load(os.path.join(path, 'intensity_samples_' + str(channel) + '.npy'))
        else:
            intensity_samples = sampleIntensities(layer_stack)
            np.save('intensity_samples_'+str(channel)+'.npy', intensity_samples)

        # Compute Response Curve
        response_curve = computeResponseCurve(intensity_samples, log_exposure_times, smoothing_lambda, Weight)

        # Build radiance map
        img_rad_map, place = computeRadianceMap(layer_stack, log_exposure_times, response_curve, eps)
        places.append(place)

        # Normalize hdr layer to (0, 255)
        hdr_image[..., channel] = cv2.normalize(img_rad_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # hdr_image[..., channel] =255. * (img_rad_map - img_rad_map.min())/(img_rad_map.max() - img_rad_map.min())

    # Global tone mapping
    image_mapped = globalToneMapping(hdr_image, gamma)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 15))
    # for i in range(1, 21):
    #     plt.subplot(4, 5, i)
    #     plt.imshow(globalToneMapping(hdr_image, 0.05 * i))

    # Adjust image intensity based on the middle image from image stack
    num = len(images)//2 - (log_exposure_times[0] > log_exposure_times[1]) and len(images) % 2 == 0
    template = images[num]
    image_tuned = intensityAdjustment(image_mapped, template)

    # make output image
    # Notice: this method min_max all value in all channels at the same time not for everyone
    output = cv2.normalize(image_tuned, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # brightness and contrast adjusting
    # brightness = 0
    # contrast = 0
    # factor = (259 * (255 + contrast)) / (255 * (259 - contrast))
    # f = np.arange(256)
    # f[eps:255 - eps] = factor * (f[eps:255 - eps] - 128) + 128
    # f[eps:int(128 - (128 - eps) / factor)] = eps
    # f[int(128 + (128 - eps) / factor):255 - eps] = 255 - eps
    # f = f.astype(np.uint8)
    # output += brightness
    # output[output < 0] = 0
    # output[output > 255] = 255
    # output = f[output.astype(np.uint8)]

    # recover eps part
    max_exposure = np.argmax(log_exposure_times)
    min_exposure = np.argmin(log_exposure_times)
    for channel in range(num_channels):
        output[..., channel][places[channel][0]] = images[min_exposure][..., channel][places[channel][0]]
        output[..., channel][places[channel][1]] = images[max_exposure][..., channel][places[channel][1]]

    # output[..., 0] = globalToneMapping(output[..., 0], gamma=0.92) * 255
    return output.astype(np.uint8)


if __name__ == "__main__":
    from PIL import Image
    import time
    images = []
    for i in [0, 1]:
        name = r'D:\Project\pycharm\Camera\test\20171001114000_1' + str(i) + '.jpg'
        print(name)
        image = Image.open(name)
        image = np.array(image)
        images.append(image[:, :, :3])
    log_exposure_times = np.array([log(1), log(2)])
    start = time.time()
    # images = [img[350:2350,:,:] for img in images]
    path = r'D:\Project\pycharm\Camera\test\\'
    o = computeHDR(images, log_exposure_times, state='save', path=path, smoothing_lambda=100, gamma=0.6, eps=40)
    print('Time', time.time() - start)
    o = Image.fromarray(o)
    o.save(str(i)+'.png')