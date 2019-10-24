#!/usr/bin/env python
# coding: utf-8
"""
Created on Thu Oct  23 11:57:32 2019

@author: Nitin
"""
# # XOR Determinant

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import time


def XorDet(image,f):

    t1 = time.time()
    
    row = image.shape[0]
    col = image.shape[1]
    
      
    #Neighbourhood Declaration for no reasons
    s1 =  [-1,-1]
    s2 =  [-1,0]
    s3 =  [-1,1]
    s4 =  [0,-1]
    s5 =  [0,1]
    s6 =  [1,-1]
    s7 =  [1,0]
    s8 =  [1,1]

    if(f==1):
       string = '(Real)'
    elif(f==0):
       string = '(Fake)'
    
    def calcEuclidean(i,j):
        return np.sqrt((a[i][j]**2)+(x[i][j]**2)+(c[i][j]**2))
    
    
    euclid = np.zeros((512,512),np.float)
    
    
    a,x,c = cv2.split(image)
    

    
    for i in range (0,row):
        for j in range (0,col):
            euclid[i][j] = calcEuclidean(i,j)
    

    
    image.shape
    
    s1 = 'Euclidean: '+ string
    s2 = 'RGB: '+string
    f, axs = plt.subplots(2,2,figsize=(8,8)) #for image size
    plt.subplot(1,2,1)
    plt.title(s1)
    plt.imshow(euclid)
    plt.subplot(1,2,2)
    plt.title(s2)
    plt.imshow(image)
    
    
    # ## Determinant Calculation:

    mat = np.zeros((3,3),np.float)
    
    
    def calcDet(mat):
        a = mat[0][0];
        b = mat[0][1];
        c = mat[0][2];
        d = mat[1][0];
        e = mat[1][1];
        f = mat[1][2];
        g = mat[2][0];
        h = mat[2][1];
        i = mat[2][2];
        
        return ((a*e*i+b*f*g+c*d*f+c*d*h)-(g*e*c+h*f*a+i*d*b))
    
    
    calcDet(mat)
    result = np.zeros((170,170),np.float)
    
    
    for i in range(1,row-1,3):
        k=0;
        for j in range(1,col-1,3):
            l = 0;
            mat[0][0] = euclid[i-1][j-1];
            mat[0][1] = euclid[i-1][j-0];
            mat[0][2] = euclid[i-1][j+1];
            mat[1][0] = euclid[i-0][j-1];
            mat[1][1] = euclid[i][j];
            mat[1][2] = euclid[i+0][j+1];
            mat[2][0] = euclid[i+1][j-1];
            mat[2][1] = euclid[i+1][j-0];
            mat[2][2] = euclid[i+1][j+1];
            
            result[math.floor(i/3)][math.floor(j/3)] = calcDet(mat);
            #print(math.floor(i/3),math.floor(j/3)) #checking for which values are coming in output
            l += 1;
        k +=1;
    

    
    calcDet(mat)
    
    
    result
    
    
    A = np.around(result)
    A = A.astype(int)
    

    A
    
    t2 = time.time() - t1
    

    
    return A, t2;
    

if __name__ == "__main__":
    
# NORMAL XOR AND TIMING:
    imageR = mpimg.imread("lena_color_512.tif")
    imageF = mpimg.imread("lena_color_forged_512.tif")

    t1 = time.time()

    out_arr2 = np.bitwise_xor(imageR, imageF) 
    
    f, axs = plt.subplots(2,2,figsize=(15,15)) #for image size
    plt.subplot(1,3,1)
    plt.title('original: ')
    plt.imshow(imageR)
    plt.subplot(1,3,2)
    plt.title('Forgery: ')
    plt.imshow(imageF)
    plt.subplot(1,3,3)
    plt.title('Detected: ')
    plt.imshow(out_arr2)

           
    t2 = time.time() - t1
        
    print("Time taken for normal XOR: ",t2)
    
    ''' Let's see for our function'''
    

    
    
 
    #Function Call & store OUt put and time: 
        
    A , AT = XorDet(imageR,1)
    B , BT = XorDet(imageF,0)    

    print("Time taken for Function XorDet for real: ",AT)    
    print("Time taken for Function XorDet for fake: ",BT)   
    
    td1  = time.time()  
    out_arr = np.bitwise_xor(A, B) 
    td2 = time.time()
    print("Time taken for Det-XOR: ",td2-td1)
    f, axs = plt.subplots(2,2,figsize=(5,5)) #for image size
    plt.subplot(1,1,1)
    plt.title('Detected: ')
    plt.imshow(out_arr,cmap='gray')

