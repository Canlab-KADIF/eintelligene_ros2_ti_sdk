#!/usr/bin/env python2.7

#  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Calibrate camera intrinsic parameters
"""
from __future__ import print_function

import sys
import os
import glob
import argparse
import cv2
import numpy as np

def parse_args():
    """Parse input argument
    """

    def str2bool(value):
        """Convert a string to boolean type"""
        if value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument( '-p', '--path', type=str, default='./demo_calib',
        help='Path to the directory storing images for calibration')
    parser.add_argument( '-e', '--ext', type=str, default='jpg',
        help='Extension of the files')
    parser.add_argument( '-r', '--pts_rows', type=int, default=7,
        help='Number of inner corners per row')
    parser.add_argument( '-c', '--pts_columns', type=int, default=6,
        help='Number of inner corners per column')
    parser.add_argument( '-s', '--square_size', type=int, default=1,
        help='Size of a square (the edge, by mm) of the checker board')
    parser.add_argument( '-v', '--visualize', type=str2bool, default=True,
        help='Visualize the checker board or not')
    parser.add_argument( '-f', '--scaling_factor', type=float, default=-1,
        help='scaling factor after reading images')
    parser.add_argument( '-o', '--out_fname', type=str, default=None,
        help='Path to the output file to store result (in YAML format)')
    parser.add_argument( '-n', '--camera_name', type=str, default='camera',
        help='Camera name')

    args = parser.parse_args()
    assert os.path.isdir(args.path), '{} not found'.format(args.path)
    return args

def natural_keys(value):
    """Convert to natural key for sorting string numbers
    """
    value = os.path.basename(value).replace('.png', '')
    if '_' in value:
        return int(value.split('_')[-1])
    return value

def read_image(fname, scaling_factor, to_gray=True):
    """Read image, resize, and convert to gray scale accordingly

    Args:
        fname: path to the image
        scaling_factor: a positive float, the scaling factor to resize the image
        to_gray: whether to convert the color image into gray scale

    Return:
        Loaded and processed image
    """
    img = cv2.imread(fname)

    if scaling_factor > 0:
        im_h, im_w, _ = img.shape
        im_h, im_w = int(scaling_factor*im_h), int(scaling_factor*im_w)
        img = cv2.resize(img, (im_w, im_h), interpolation=cv2.INTER_CUBIC)

    if to_gray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
    return img

def save_results(mtx, dist, im_w, im_h, fname, camera_name='camera'):
    """Save the calibration results as YAML format

    File structure follows the convention found at: http://wiki.ros.org/camera_calibration_parsers

    Note: Simply loading a calibration file does not rectify the image.
    (ref: http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration)
    Therefore, the rectification is just an identity matrix

    Args:
        mtx: camera matrix
        dist: distortion coefficients
        im_w, im_h: width and height of the image (after rescaling)
        fname: output filename
        camera_name: name of the camera
    """

    def __write_matrix(outfile, msg, data):
        """Wrapper to write content of a matrix to YAML file

        Args:
            outfile: initialized file handler
            msg: message to add at the beginning to describe the matrix
            data: ndarray, the matrix to print out
        """
        outfile.write('{}:\n'.format(msg))
        outfile.write('  rows: {}\n'.format(data.shape[0]))
        outfile.write('  cols: {}\n'.format(data.shape[1]))
        outfile.write('  data: {}\n'.format(data.flatten().tolist()))

    with open(fname, 'w') as outfile:
        outfile.write('image_width: {}\n'.format(im_w))
        outfile.write('image_height: {}\n'.format(im_h))
        outfile.write('camera_name: {}\n'.format(camera_name))

        __write_matrix(outfile, 'camera_matrix', mtx)

        if len(dist.flatten()) == 5:
            dist_model = 'plumb_bob'
        elif len(dist.flatten()) == 8:
            dist_model = 'rational_polynomial'
        else:
            # raise 'Unknown distortion model'
            dist_model = 'unknown'
        outfile.write('distortion_model: {}\n'.format(dist_model))
        __write_matrix(outfile, 'distortion_coefficients', dist)

        __write_matrix(outfile, 'rectification_matrix', np.eye(3))

        proj = np.hstack([mtx, np.zeros([3, 1])])
        __write_matrix(outfile, 'projection_matrix', proj)

def save_camera_info(image_size, camera_name, dist_model, K, D, R, P, file_path):
    """ save the camera info to given file_path
    """
    cv_file = cv2.FileStorage(file_path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("image_width",             image_size[0])
    cv_file.write("image_height",            image_size[1])
    cv_file.write("camera_name",             camera_name)
    cv_file.write("camera_matrix",           K)
    cv_file.write("distortion_model",        dist_model)
    cv_file.write("distortion_coefficients", D)
    cv_file.write("rectification_matrix",    R)
    cv_file.write("projection_matrix",       P)
    cv_file.release()
    print("camera_info saved into {}".format(file_path))

def calib_nonfisheye(images, args):
    """Calibration for non-fisheye images

    (ref: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html)

    Args:
        images: list of image paths
        args: parsed input arguments

    Return:
        camera_matrix: (3, 3) ndarray, intrinsic parameters of the camera
        distortion_coefficient: (3, 1) ndarray, distortion coefficients
    """
    # Retrive pattern size
    patt_sz = (args.pts_rows, args.pts_columns)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((patt_sz[0]*patt_sz[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patt_sz[0], 0:patt_sz[1]].T.reshape(-1, 2)
    objp *= args.square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Collect corners of each image
    print('Collect corners...')
    for fname in images:
        img = read_image(fname, args.scaling_factor, to_gray=False)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, patt_sz, None)
        print(os.path.basename(fname), '-->', ret)

        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if args.visualize:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, patt_sz, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(100)
    cv2.destroyAllWindows()
    assert objpoints != [], 'No corners found'

    # Calibrate
    print('Calibrating...')
    reprojection_error, camera_matrix, distortion_coefficient, \
    rotation_v, translation_v = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    return camera_matrix, distortion_coefficient

def undistort_nonfisheye(images, camera_matrix, distortion_coefficient, args):
    """Undistort nonfisheye images using the calibrated parameters

    Args:
        images: list of image paths
        camera_matrix: (3, 3) ndarray, intrinsic parameters of the camera
        distortion_coefficient: (3, 1) ndarray, distortion coefficients
    """
    for fname in images:
        img = read_image(fname, args.scaling_factor, to_gray=False)

        im_h, im_w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, distortion_coefficient, (im_w, im_h), 1, (im_w, im_h))

        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix, distortion_coefficient, None, newcameramtx, (im_w, im_h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow('img', dst)
        cv2.waitKey(100)
    cv2.destroyAllWindows()

def main():
    """Main function"""
    args = parse_args()

    # Load data
    images = glob.glob(os.path.join(args.path, '*.'+args.ext))
    assert images != [], 'Found no images'
    images.sort(key=natural_keys)

    # Calibrate
    camera_matrix, distortion_coefficient = calib_nonfisheye(images, args)
    print('Camera matrix\n', camera_matrix)
    print('Distortion coefficients\n', distortion_coefficient)

    # # Verify calibration by undistorting images
    # undistort_nonfisheye(images, camera_matrix, distortion_coefficient, args)

    # Save results
    # camera_name = args.camera_name

    if args.out_fname:
        img = read_image(images[0], args.scaling_factor, to_gray=False)
        im_h, im_w, _ = img.shape
        # save_results(camera_matrix, distortion_coefficient, im_w, im_h,
        #              args.out_fname, 'camera')

        rect_matrix = np.eye(3)
        proj_matrix = np.hstack( (camera_matrix, np.zeros((3,1))) )
        save_camera_info([im_w, im_h], 'camera', 'plumb_bob', camera_matrix,
                         distortion_coefficient, rect_matrix, proj_matrix, args.out_fname)

    return 0

if __name__ == '__main__':
    sys.exit(main())
