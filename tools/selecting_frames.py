#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:42:35 2018
@author: Hemin Ali Qadir
goal: clearing redundancy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import glob
import cv2 

parser = argparse.ArgumentParser()
parser.add_argument("--RGB_dir", required=True, help="path to the folder containing rgb images")
parser.add_argument("--GT_dir", required=True, help="path to the foloder containing the GT images")
parser.add_argument("--output_filetype", required=True, default=".png", choices=[".png", ".jpg"])
parser.add_argument("--num_frames", required=True, help="number of frames in the video")
parser.add_argument("--output_dir", required=True,  help="where to put output files")  # required=True,
a = parser.parse_args()

# check the output image type 
if (a.output_filetype != '.png'):
    if (a.output_filetype != '.jpg'):
        raise ValueError("image type is not supported, please choose either .png or .jpg")
        
def get_name(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name

#%%
# Listing the images in the Original folder and making a folder 
rgb_paths = glob.glob(os.path.join(a.RGB_dir, "*.jpg"))
if len(rgb_paths) == 0:
    rgb_paths = glob.glob(os.path.join(a.RGB_dir, "*.png"))
if len(rgb_paths) == 0:
    raise Exception("RGB_dir contains no image files")
# if the image names are numbers, sort by the value rather than asciibetically
# having sorted inputs means that the outputs are sorted in test mode
if all(get_name(path).isdigit() for path in rgb_paths):
    rgb_images = sorted(rgb_paths, key=lambda path: int(get_name(path)))
else:
    rgb_images = sorted(rgb_paths)

if not os.path.exists(a.output_dir+'/Original'):
    os.makedirs(a.output_dir+'/Original')

#%%
# Listing all the images in the GT folder and make a folder 
gt_paths = glob.glob(os.path.join(a.GT_dir, "*.jpg"))
if len(gt_paths) == 0:
    gt_paths = glob.glob(os.path.join(a.GT_dir, "*.png"))
if len(gt_paths) == 0:
    raise Exception("GT_dir contains no image files")
if all(get_name(path).isdigit() for path in gt_paths):
    gt_images = sorted(gt_paths, key=lambda path: int(get_name(path)))
else:
    gt_images = sorted(gt_paths)
    
if not os.path.exists(a.output_dir+'/GT'):
    os.makedirs(a.output_dir+'/GT')

#%%
# Save the first and the last frames all the time 
list_of_frames = [x for x in range(0, int(a.num_frames), 25)] + [int(a.num_frames)-1]

for i in list_of_frames:
    
    print('saving frame: {}'.format(i))
    # read in the images 
    rgb_image = cv2.imread(rgb_images[i])
    gt_image  =cv2.imread(gt_images[i])
    
    # store the target images 
    cv2.imwrite(os.path.join(a.output_dir, 'Original', str(i) + a.output_filetype), rgb_image)
    cv2.imwrite(os.path.join(a.output_dir, 'GT', str(i) + a.output_filetype), gt_image)
    
    
