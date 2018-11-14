#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 23:18:53 2018

@author: Hemin Ali Qadir

@process: Data Augemntaion 
         - Rotaion
         - shearing
         - bluring
         - zoom in
         - zoom out
         - fliping 
"""
#%% 

import math
import cv2
import os, os.path 
import numpy as np 
from skimage import transform as tf
from skimage import img_as_ubyte
from PIL import ImageEnhance
from PIL import Image

# A function to make a directory if it does not exsit
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
# Shearing using opencv crop the images to the same size of the original image
# Therefore we need to pad zeros to the borders 
# padding zeros to the borders just for our images 
def padzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

def pad0img(img,scale, col_dif, row):
    
    b, g, r = cv2.split(img)
    b = np.lib.pad(b, col_dif, padzeros)
    g = np.lib.pad(g, col_dif, padzeros)
    r = np.lib.pad(r, col_dif, padzeros)
    img1 = cv2.merge((b,g,r))
    row_n, col_n, ch_n = img1.shape
    new_row = int((row_n - row)/2)
    img1 = img1[new_row:-new_row,:,:] 
        
    return img1

# Adding Gaussian Noise 
def noisy(image, var):
    # mean = 0
    row,col,ch= image.shape
    
    sigma = var**0.5
    
    # Adding Gaussian noise 
    # Two ways of doing this opencv or numpy 
    # noise = np.zeros((image.shape), np.int8)
    # cv2.randn(noise, np.zeros(3), np.ones(3)*255*sigma)
    noise = (sigma*255) * np.random.randn(*image.shape)
    noisy_image = cv2.add(image, noise, dtype=cv2.CV_8UC3)
    
    return noisy_image

def image_augmentation_blurring(load_img_fold,load_gt_fold, store_img_fold, store_gt_fold, blur_factor):
    
    make_dir(store_img_fold)
    make_dir(store_gt_fold)
    
    orig_image = os.listdir(load_img_fold)
    # gt_image = os.listdir(load_gt_fold)      # images have the same name
    
    for i in orig_image :
        
        # Orignal image 
        orig_img = cv2.imread(load_img_fold + '/' + i)
        
        # I get this formula from the MATLAB tourial 
        ker_size = 2*math.ceil(2*blur_factor)+1
        
        orig_img_blur = cv2.GaussianBlur(orig_img,(int(ker_size),int(ker_size)),blur_factor)
        
        cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_blur'  + i[-4:], orig_img_blur)
        
        # GT image 
        gt_img = cv2.imread(load_gt_fold + '/' + i)
        
        # gt_img_blur = cv2.GaussianBlur(gt_img,(ker_size,ker_size),blur)
        
        cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_blur'  + i[-4:], gt_img)
        
# Image Zoom in 
def image_augmentation_zoom_in(load_img_fold,load_gt_fold, store_img_fold, store_gt_fold, zoom_rate):
    
    make_dir(store_img_fold)
    make_dir(store_gt_fold)
    
    orig_image = os.listdir(load_img_fold)
    # gt_image = os.listdir(load_gt_fold)      # images have the same name
    
    for i in orig_image:
        for j in zoom_rate:
            
            # Orignal image 
            orig_img = cv2.imread(load_img_fold + '/' + i);
            gt_img = cv2.imread(load_gt_fold + '/' + i)#[:-4]+'_mask.tif')
            thresh = 127
            gt_img = cv2.threshold(gt_img, thresh, 255, cv2.THRESH_BINARY)[1]/255
            gt_size = np.sum(gt_img)
            
            # Orginal and GT images have the same size
            row, col, ch = orig_img.shape
            row_factor = int(math.ceil(row*j));
            col_factor = int(math.ceil(col*j));
            
            img_zoom_org = orig_img[col_factor:col_factor+(row-row_factor*2-1), row_factor:row_factor+(col-col_factor*2-1)]
            img_zoom_orig = cv2.resize(img_zoom_org, ((col), (row)), interpolation=cv2.INTER_CUBIC)
            
            
            img_zoom_gtc = gt_img[col_factor:col_factor+(row-row_factor*2-1), row_factor:row_factor+(col-col_factor*2-1)]
            img_zoom_gt = cv2.resize(img_zoom_gtc, ((col), (row)), interpolation=cv2.INTER_CUBIC)
            
            if np.sum(img_zoom_gt)>=gt_size:
                cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_in_'  + str(j) + i[-4:], img_zoom_gt*255)
                cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_in_'  + str(j) + i[-4:], img_zoom_orig)
    
def image_augmentation_zoom_out(load_img_fold,load_gt_fold, store_img_fold, store_gt_fold, zoom_rate):
    
    make_dir(store_img_fold)
    make_dir(store_gt_fold)
    
    orig_image = os.listdir(load_img_fold)
    # gt_image = os.listdir(load_gt_fold)      # images have the same name
    
    for i in orig_image:
        for j in zoom_rate:
            
            # Orignal image 
            orig_img = cv2.imread(load_img_fold + '/' + i);
            gt_img = cv2.imread(load_gt_fold + '/' + i)#[:-4]+'_mask.tif')

            row, col, ch = orig_img.shape
            row_factor = row*j;
            col_factor = col*j;
            img_zoom_org = cv2.resize(img_as_ubyte(orig_img), (int(math.ceil(col-col_factor*2)),int(math.ceil(row-row_factor*2))), interpolation=cv2.INTER_CUBIC)
            
            row_out, col_out, ch_out = img_zoom_org.shape
            col_dif = int((col - col_out)/2)
            img_zoom_orig = pad0img(img_zoom_org,j, col_dif, row)
            
            cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_out_'  + str(j) + i[-4:], img_zoom_orig)
            img_zoom_gtt = cv2.resize(img_as_ubyte(gt_img), (int(math.ceil(col-col_factor*2)),int(math.ceil(row-row_factor*2))), interpolation=cv2.INTER_CUBIC);
            img_zoom_gt = pad0img(img_zoom_gtt,j, col_dif, row)
            cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_out_'  + str(j) + i[-4:], img_zoom_gt)
            
      
def image_augmentation_rotation(load_img_fold,load_gt_fold, store_img_fold, store_gt_fold):
    
    make_dir(store_img_fold)
    make_dir(store_gt_fold)
    
    orig_image = os.listdir(load_img_fold)
    # gt_image = os.listdir(load_gt_fold)      # images have the same name
    
    for i in orig_image:
        
        # Orignal image 
        orig_img = cv2.imread(load_img_fold + '/' + i);
        
        # Orginal and GT images have the same size
        rows, cols, chs = orig_img.shape
        
        # 90 degree rotation 
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        rot90 = cv2.warpAffine(orig_img, M, (cols,rows))
        # 180 degree rotaion rotation 
        M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        rot180 = cv2.warpAffine(orig_img, M, (cols,rows))
        # 270degree rotation 
        M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
        rot270 = cv2.warpAffine(orig_img, M, (cols,rows))
        # Veritical flipinh 
        vf_img=cv2.flip(orig_img,0)  
        # Horizonatal fliping 
        hf_img=cv2.flip(orig_img,1)
        
        # Back ground 
        gt_im = cv2.imread(load_gt_fold + '/' + i)#[:-4]+'_mask.tif')
        thresh = 127
        gt_img = cv2.threshold(gt_im, thresh, 255, cv2.THRESH_BINARY)[1]/255
        gt_size = np.sum(gt_img)
        # 90 degree rotation 
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        rot90_gt = cv2.warpAffine(gt_img, M, (cols,rows))
        # 180 degree rotaion 
        M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        rot180_gt = cv2.warpAffine(gt_img, M, (cols,rows))
        # 270 degree rotaion 
        M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
        rot270_gt = cv2.warpAffine(gt_img, M, (cols,rows))
        
        # Veritical fliping 
         # Veritical flipinh 
        vf_img_gt=cv2.flip(gt_img,0)  
        # Horizonatal fliping 
        hf_img_gt=cv2.flip(gt_img,1)
        
        if np.sum(rot90_gt)>=gt_size/1.5:
            # Saving the rotated images 
            cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_rot_90'  + i[-4:], rot90_gt*255)
            cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_rot_90'  + i[-4:], rot90)
        
        if np.sum(rot180_gt)>=gt_size/1.75:
            cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_rot_180' + i[-4:], rot180_gt*255)
            cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_rot_180' + i[-4:], rot180) 
        
        if np.sum(rot270_gt)>=gt_size/1.75:
            cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_rot_270' + i[-4:], rot270_gt*255) 
            cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_rot_270' + i[-4:], rot270) 
        
        if np.sum(vf_img_gt)>=gt_size/1.75:
            cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_rot_vf'  + i[-4:], vf_img_gt*255) 
            cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_rot_vf'  + i[-4:], vf_img) 
        
        if np.sum(hf_img_gt)>=gt_size/1.75:
            cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_rot_hf'  + i[-4:], hf_img_gt*255) 
            cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_rot_hf'  + i[-4:], hf_img) 

        
def image_augmentation_shearing(load_img_fold,load_gt_fold, store_img_fold, store_gt_fold):
    
    make_dir(store_img_fold)
    make_dir(store_gt_fold)
    
    orig_image = os.listdir(load_img_fold)
    # gt_image = os.listdir(load_gt_fold)      # images have the same name
    
    for i in orig_image:
        
        # Orignal image 
        orig_img = cv2.imread(load_img_fold + '/' + i)
        # GT images 
        gt_img = cv2.imread(load_gt_fold + '/' + i)#[:-4]+'_mask.tif')
        thresh = 127
        gt_img = cv2.threshold(gt_img, thresh, 255, cv2.THRESH_BINARY)[1]/255
        gt_size = np.sum(gt_img)
        
        # Orginal and GT images have the same size
        row, col, ch = orig_img.shape
        
        img = orig_img #cv2.copyMakeBorder(orig_img,350,350,250,200,cv2.BORDER_CONSTANT,value=0)
        
        matrix1 = np.array([[1,0,0],[0.5,1,0],[0,0,1]])
        matrix2 = np.array([[1,0,0],[-0.5,1,0],[0,0,1]])
        matrix3 = np.array([[1,0.5,0],[0,1,0],[0,0,1]])
        matrix4 = np.array([[1,-0.5,0],[0,1,0],[0,0,1]])
        
        afine_tf1 = tf.AffineTransform(matrix=matrix1)
        afine_tf2 = tf.AffineTransform(matrix=matrix2)
        afine_tf3 = tf.AffineTransform(matrix=matrix3)
        afine_tf4 = tf.AffineTransform(matrix=matrix4)
        
        # Apply transform to image data
        imgm1 = tf.warp(img, afine_tf1, clip=True)
        imgm11 = imgm1#[12:382,220:570]#[12:382,220:570]
        imgm11 = cv2.resize(img_as_ubyte(imgm11) ,(col, row), interpolation = cv2.INTER_CUBIC)
        
        imgm2 = tf.warp(img, afine_tf2, clip=True)
        imgm22 = imgm2#[410:780,220:570]
        imgm22 = cv2.resize(img_as_ubyte(imgm22) ,(col, row), interpolation = cv2.INTER_CUBIC)
        
        imgm3 = tf.warp(img, afine_tf3, clip=True)
        imgm33 = imgm3#[225:575,15:385]
        imgm33 = cv2.resize(img_as_ubyte(imgm33) ,(col, row), interpolation = cv2.INTER_CUBIC)
        
        imgm4 = tf.warp(img, afine_tf4, clip=True)
        imgm44 = imgm4#[225:575,414:784]
        imgm44 = cv2.resize(img_as_ubyte(imgm44) ,(col, row), interpolation = cv2.INTER_CUBIC)
        

        img_gt = gt_img #cv2.copyMakeBorder(gt_img,250,250,200,200,cv2.BORDER_CONSTANT,value=0)
        
        # Apply transform to image data
        imgt1 = tf.warp(img_gt, afine_tf1, clip=True)
        imgt11 = imgt1#[12:382,220:570]
        imgt11 = cv2.resize(img_as_ubyte(imgt11) ,(col, row), interpolation = cv2.INTER_CUBIC)
        
        imgt2 = tf.warp(img_gt, afine_tf2, clip=True)
        imgt22 = imgt2#[410:780,220:570]
        imgt22 = cv2.resize(img_as_ubyte(imgt22) ,(col, row), interpolation = cv2.INTER_CUBIC)
        
        imgt3 = tf.warp(img_gt, afine_tf3, clip=True)
        imgt33 = imgt3#[225:575,15:385]
        imgt33 = cv2.resize(img_as_ubyte(imgt33) ,(col, row), interpolation = cv2.INTER_CUBIC)
        
        imgt4 = tf.warp(img_gt, afine_tf4, clip=True)
        imgt44 = imgt4#[225:575,414:784]
        imgt44 = cv2.resize(img_as_ubyte(imgt44) ,(col, row), interpolation = cv2.INTER_CUBIC)
        
        # Saving the rotated images 
        if np.sum(imgt11)>=(gt_size/1.5):
            cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_shear_x'  + i[-4:], (imgt11*255))
            cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_shear_x'  + i[-4:], (imgm11))
        else:
            pass
        if np.sum(imgt22)>=(gt_size/1.5):        
            cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_shear_y'  + i[-4:], (imgt22*255))
            cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_shear_y'  + i[-4:], (imgm22))
        else:
            pass
          
        if np.sum(imgt33)>=(gt_size/1.5):
            cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_shear_xx'  + i[-4:], (imgt33*255))
            cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_shear_xx'  + i[-4:], (imgm33))
        else:
            pass
        
        if np.sum(imgt44)>=(gt_size/1.5):
            cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_shear_yy'  + i[-4:], (imgt44*255))
            cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_shear_yy'  + i[-4:], (imgm44))
        else:
            pass


def image_augmentation_noise(load_img_fold,load_gt_fold, store_img_fold, store_gt_fold,noise_rate):
    
    make_dir(store_img_fold)
    make_dir(store_gt_fold)
    
    orig_image = os.listdir(load_img_fold)
    # gt_image = os.listdir(load_gt_fold)      # images have the same name
    
    for i in orig_image:
        for j in noise_rate:
            
            # Orignal image 
            orig_img = cv2.imread(load_img_fold + '/' + i)
            gt_img = cv2.imread(load_gt_fold + '/' + i)
            
            nosie_image = noisy(orig_img, j)
            
            cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_noise_' + str(j) + i[-4:], nosie_image)
            
            cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_noise_' + str(j) + i[-4:], gt_img)
            

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def image_augmentation_brighten(load_img_fold,load_gt_fold, store_img_fold, store_gt_fold, bright_rate):
    
    make_dir(store_img_fold)
    make_dir(store_gt_fold)
    
    orig_image = os.listdir(load_img_fold)
    # gt_image = os.listdir(load_gt_fold)      # images have the same name
    
    for i in orig_image:
            
            # Orignal image 
            orig_img = cv2.imread(load_img_fold + '/' + i)
            gt_img = cv2.imread(load_gt_fold + '/' + i)
            
            for j in bright_rate:
                
                bright_image = adjust_gamma(orig_img, gamma=j)
            
                cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_bright_' + str(j) + i[-4:], bright_image)
            
                cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_bright_' + str(j) + i[-4:], gt_img)
            

def image_augmentation_darken(load_img_fold,load_gt_fold, store_img_fold, store_gt_fold, darken_rate):

    make_dir(store_img_fold)
    make_dir(store_gt_fold)
    
    orig_image = os.listdir(load_img_fold)
    # gt_image = os.listdir(load_gt_fold)      # images have the same name
    
    for i in orig_image:
        img = Image.open(load_img_fold + '/' + i)
        gt_img = cv2.imread(load_gt_fold + '/' + i)
        
        enhancer = ImageEnhance.Brightness(img)
        
        for j in darken_rate:

            darken_image = np.array(enhancer.enhance(j))
            
            darken_image = cv2.cvtColor(darken_image, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(store_img_fold + '/' + i[0:-4] + '_dark_' + str(j) + i[-4:], darken_image)
            cv2.imwrite(store_gt_fold + '/' + i[0:-4] + '_dark_' + str(j) + i[-4:], gt_img)

from distutils.dir_util import copy_tree
           
# the python will look to main function and implemented first 
if __name__ == "__main__":
    
    load_dir = '/home/hemin/Desktop/Semi_Automatic_OSVOS/Video_1/per20_correct_aug/Training_Set'
    load_dir_org = load_dir+"/Original_16"  
    load_dir_gt = load_dir+"/GT_16"  
        
#    load_dir_org = "/home/hemin/Desktop/Polyp_Segmentation/SD/Training/Augmentation/Rotation/RGB" 
#    load_dir_gt  = "/home/hemin/Desktop/Polyp_Segmentation/SD/Training/Augmentation/Rotation/GT" 
    save_dir = '/home/hemin/Desktop/Semi_Automatic_OSVOS/Video_1/per20_correct_aug/Training_Set/darken'
    save_dir_org = save_dir + "/Original" 
    save_dir_gt = save_dir + "/GT" 

#    copy_tree(save_dir_org, load_dir_org)
#    copy_tree(save_dir_gt, load_dir_gt)
    
#    image_augmentation_rotation(load_dir_org, load_dir_gt, save_dir_org, save_dir_gt)
    
#    image_augmentation_zoom_in(load_dir_org,load_dir_gt, save_dir_org, save_dir_gt, np.array([0.1,0.2]))
    
#    image_augmentation_zoom_out(load_dir_org,load_dir_gt, save_dir_org, save_dir_gt, np.array([0.1,0.2,0.3]))
    
#    image_augmentation_blurring(load_dir_org,load_dir_gt, save_dir_org, save_dir_gt, 1.3)
    
#    image_augmentation_noise(load_dir_org,load_dir_gt, save_dir_org, save_dir_gt,np.array([0.001,0.002]))

    image_augmentation_darken(load_dir_org,load_dir_gt, save_dir_org, save_dir_gt, np.array([0.5,0.75]))
    
#    image_augmentation_brighten(load_dir_org,load_dir_gt, save_dir_org, save_dir_gt, np.array([1.5,1.25]))