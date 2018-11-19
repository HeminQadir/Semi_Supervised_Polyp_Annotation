#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:06:22 2018

@author: hemin

Goal: to change image types to png or jpg
"""
import cv2
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--RGB_dir", required=True, help="path to the folder containing rgb images")
parser.add_argument("--GT_dir", required=True , help="path to the foloder containing the GT images")
parser.add_argument("--output_filetype", default=".png", choices=[".png", ".jpg"])
parser.add_argument("--output_dir", default='kk',  help="where to put output files")  # required=True,
a = parser.parse_args()

def get_name(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name

# check the output image type 
if (a.output_filetype != '.png'):
    if (a.output_filetype != '.jpg'):
        raise ValueError("image type is not supported, please choose either .png or .jpg")

# make the output directory if not exists  
if not os.path.exists(a.output_dir):
    os.makedirs(a.output_dir)

# make the output directory if not exists  
if not os.path.exists(a.output_dir+'/Original'):
    os.makedirs(a.output_dir+'/Original')
    
# make the output directory if not exists  
if not os.path.exists(a.output_dir+'/GT'):
    os.makedirs(a.output_dir+'/GT')
    
# Read the rgb image names 
rgb_images = os.listdir(a.RGB_dir)
# if the image names are numbers, sort by the value rather than asciibetically
# having sorted inputs means that the outputs are sorted in test mode
if all(get_name(path).isdigit() for path in rgb_images):
    rgb_images = sorted(rgb_images, key=lambda path: int(get_name(path)))
else:
    rgb_images = sorted(rgb_images, key = lambda x: int(x[2:].split(".")[0]))
    
j = 1
for i in rgb_images:
    print('change image number: {}'.format(j))
    rgb_image = cv2.imread(os.path.join(a.RGB_dir,i))
    
    cv2.imwrite(os.path.join(a.output_dir, 'Original', str(j) + a.output_filetype), rgb_image)
    
    j +=1
    
# read the GT image names 
gt_images = os.listdir(a.GT_dir)
# if the image names are numbers, sort by the value rather than asciibetically
# having sorted inputs means that the outputs are sorted in test mode
if all(get_name(path).isdigit() for path in rgb_images):
    # the lambda function has to be replace for other set of dataset. mine is 'im#.tiff'
    gt_images = sorted(gt_images, key=lambda path: int(get_name(path)))
else:
    # the lambda function has to be replace for other set of dataset. mine is 'p#.tiff'
    gt_images = sorted(gt_images, key = lambda x: int(x[1:].split(".")[0]))
j = 1
for i in gt_images:
    print('change image number: {}'.format(j))
    gt_image = cv2.imread(os.path.join(a.GT_dir,i))
    
    cv2.imwrite(os.path.join(a.output_dir, 'GT', str(j) + a.output_filetype), gt_image)
    
    j +=1