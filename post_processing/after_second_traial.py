#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 09:32:37 2018

@author: hemin
Goal: Run this file to eliminate FP using Fourier decipitor 
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from metric import jaccard, dice 
from fourier_decipitor import (elliptic_fourier_descriptors, 
                               separate_objects, 
                               center_cx_cy,
                               compute_area)
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os 

#%%
path_to_image = '/home/hemin/Desktop/Annoataion_Work/Each_Video/Video_1/second_trail/Results/25/images'
path_to_save = '/media/hemin/Data/Semi_Auto/Collecting_Method/Similar_Masks'
howmany_frames = 324


ref_images = [o+1 if o is 0 else o for o in range(0,howmany_frames,25)]+[howmany_frames]
plt.ion()
both_sides = range(1, howmany_frames+1) + range(howmany_frames, 0, -1)

for k in both_sides:  

    print('image: {}'.format(k))
    if k in ref_images: 
        print('This image is annoated by a doctor')
        # Read in the refrence images 
        ref_image = cv2.imread(os.path.join(path_to_image,str(k)+'-targets.png'), 0)
        ret, ref_image = cv2.threshold(ref_image, 127, 255,cv2.THRESH_BINARY)
        # You have to save the refrence images all the time 
        cv2.imwrite(os.path.join(path_to_save,'GT', str(k)+'-outputs.png'), ref_image) 
        cv2.imwrite(os.path.join(path_to_save,'Original', str(k)+'.png'),
                    cv2.imread(os.path.join(path_to_image,str(k)+'-inputs.png')))
    
    # Separate the objects 
    ref_objects = separate_objects(ref_image)
    
    # Read in the query images 
    query_image = cv2.imread(os.path.join(path_to_image,str(k)+'-outputs.png'), 0)
    ret, query_image = cv2.threshold(query_image, 127, 255,cv2.THRESH_BINARY)
    query_objects = separate_objects(query_image)
    
    # Show the similar objects 
    similar_object = np.zeros(ref_image.shape, dtype=np.uint8)
    
    for ref_object in ref_objects: 
        for query_object in query_objects:
            # This is only added to aviod empty objects 
            if np.sum(ref_object)*np.sum(query_object) >0:
                # Find the contours of the refrence image using OpenCV.
                _, cnt_ref, hierarchy = cv2.findContours(ref_object, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
                # Find the coefficients of all contours
                coeffs_ref = elliptic_fourier_descriptors(np.squeeze(cnt_ref), order=1000, normalize=True)    
                # Find the contours of the query image using OpenCV.
                _, cnt_query, hierarchy = cv2.findContours(query_object, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
                coeffs_query = elliptic_fourier_descriptors(np.squeeze(cnt_query), order=1000, normalize=True)
        
                # Compute L1 distance to measure the similarity between the two objects
                L1_dst = np.sum(abs(coeffs_ref-coeffs_query))
                center_dst = np.sum(abs( center_cx_cy(ref_object) - center_cx_cy(query_object) ))
                
                area_ref = compute_area(ref_object)
                area_query = compute_area(query_object)
                # computing the area ration 
                if area_ref < area_query:
                    area_ratio = area_ref/float(area_query)
                else:
                    area_ratio = area_query/float(area_ref)
                
                print ("how similary the shapes: {0:0.5f}, distance: {1:0.5f}, area_ratio: {2:.5f}".format(L1_dst, center_dst, area_ratio))
                
                if L1_dst <=1.5 and center_dst<75 and area_ratio >0.5:
                    print('They are simialr')
                    # add the similar objects in an image 
                    similar_object = cv2.bitwise_or(similar_object, query_object)
                    
                    # We do not need to save the refrence images again 
                    if k not in ref_images:
                        cv2.imwrite(os.path.join(path_to_save,'GT', str(k)+'-outputs.png'), similar_object*255) 
                        cv2.imwrite(os.path.join(path_to_save,'Original', str(k)+'.png'),
                                    cv2.imread(os.path.join(path_to_image,str(k)+'-inputs.png')))
                        
                    ref_image = similar_object*255
            # How about if there is no objects 
            else:
                pass 
            # Thinking to remove the objects in the predicted mask 
            
                
    print('-'*50)
    
#%% Computing the perfromance 

result_dice = []
result_jaccard = []

for im in range(1,howmany_frames):
    
    # Read in the refrence images 
    input_GT = cv2.imread(os.path.join(path_to_image,str(im)+'-targets.png'), 0)
    ret, input_GT = cv2.threshold(input_GT, 127, 255,cv2.THRESH_BINARY)
    
    output_GT = cv2.imread(os.path.join(path_to_save,'GT', str(im)+'-outputs.png'), 0)
    if output_GT is not None:
        ret, output_GT = cv2.threshold(output_GT, 127, 255,cv2.THRESH_BINARY)
    else:
        output_GT = np.zeros(input_GT.shape, dtype=np.uint8)
        cv2.imwrite(os.path.join(path_to_save,'GT', str(im)+'-outputs.png'), output_GT) 
        
    # Segemnation Evaluation 
    result_dice += [dice((input_GT>255*0.5).astype(np.uint8), (output_GT>255*0.5).astype(np.uint8))]
    result_jaccard += [jaccard((input_GT>255*0.5).astype(np.uint8), (output_GT>255*0.5).astype(np.uint8))]
    
        
print('')
print('#'*80)
print('Summery of Segemantion Evaluation')
print('Dice mean: {:.5}           Dice std: {:.5}'.format(str(np.mean(result_dice)), str(np.std(result_dice))))
print('Jaccard mean: {:.5}        Jaccard std:  {:.5}'.format(str(np.mean(result_jaccard)), str(np.std(result_jaccard))))
print('#'*80)
print('')
    