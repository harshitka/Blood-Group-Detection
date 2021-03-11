from cv2 import *
import numpy as np
from skimage.io import imread, imshow
from skimage.util import random_noise
from matplotlib import pyplot as plt
from PIL import Image  
#from scipy.fft import fft, ifft
import scipy
from scipy import signal
from math import log10, sqrt 
import sys
import math


def RGB2HSV(I):
    hsv = cv2.cvtColor(I, cv2.COLOR_RGB2HSV)
    return hsv

def RGB2GRAY(I):
    gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    return gray

def binary_fun(I):
    bin=cv2.threshold(I, 127, 255, cv2.THRESH_BINARY)[1]
    return bin

def Hist1(image):
    list=np.zeros((1000))
    #cv2.imshow('image',image)
    r=image.shape[0]
    c=image.shape[1]
    # print(image.shape)
    for i1 in range(c):
        for j1 in range(r):
            if(image[j1,i1]==255):
                list[j1]=list[j1]+1

    plt.hist(list)

    return 
def cluster(image):
    pic = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel=np.zeros((3,5))
    for i in range(3):
        for j in range(5):
            kernel[i][j]=1
    
    pic = cv2.erode(pic, kernel, iterations=2)
    para = cv2.SimpleBlobDetector_Params()
    
    para.filterByConvexity = True
    para.minConvexity = 1/1000
    
    para.filterByInertia = True
    para.minInertiaRatio = 1/1000

    para.filterByArea = True
    para.minArea = 2
    

    
    para.filterByColor = True
    para.blobColor = 255
    
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(para)
    else:
        detector = cv2.SimpleBlobDetector_create(para)
    key = detector.detect(pic)
    if(len(key) <= 6):
        return 0
    else: 
        return 1


def celldensity(im):
    
    im = cv2.resize(im, (300, 300))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im=binary_fun(im)

    count_white=0
    r=im.shape[0]
    c=im.shape[1]
    i=0
    j=0
    while(i < r):
        while(j <c):
            if(im[i][j]==255):
                count_white=count_white+1
            j=j+1
        i=i+1

    if(count_white<14000):
        return 1
    else:
        return 0


def Pre(i11):
    hsv = RGB2HSV(i11)
    hsv_list=cv2.split(hsv)
    hsv = RGB2HSV(i11)
    v = RGB2GRAY(hsv)
    binary = binary_fun(v)

     
    kernel = np.zeros((15, 15), np.uint8)
    i=0
    j=0
    while(i<15):
        while(j<15):
            kernel[i][j]=1
            j=j+1
        i=i+1
    
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=2)
    return dilation


def d_e(i1):

    hsv = cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(hsv, 127, 255, cv2.THRESH_BINARY)[1]

    # Kernel 
    kernel = np.zeros((15, 15), np.uint8)

    for i in range(15):
        for j in range(15):
            kernel[i][j]=1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=2)
    #cv2.imshow('d',dilation)
    return erosion,dilation


def Seg(i1):

    hsv = cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(hsv, 127, 255, cv2.THRESH_BINARY)[1]

    # Kernel 
    kernel = np.zeros((15, 15), np.uint8)

    for i in range(15):
        for j in range(15):
            kernel[i][j]=1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=2)

    Hist1(dilation)
    c= Canny(dilation,10,250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(c, cv2.MORPH_CLOSE, kernel)

#y - 5:y + h + 5, x - 15:x + w + 5
#final = cv2.drawContours(img_copy, contours, contourIdx = -1, color = (255, 0, 0), thickness = 2)

    a=0
    b=0
    (cnts,xx) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    so= np.ones((4))
    for i in range(4):
        so[i]=0
    ox=np.ones((3))
    for i in range(3):
        ox[i]=0
    cx=np.ones((3))
    for i in range(3):
        cx[i]=0
    os=np.ones((3,4))
    for i in range(3):
        for j in range(4):
            os[i][j]=0
    true=True
    contour= sorted(cnts, key=cv2.contourArea, reverse=True)



    counter=0
    while(counter < len(contour)):
        if(a<3):
            rect_list=cv2.boundingRect(contour[counter])
            cx[a]=rect_list[0]
        a=a+1
        counter=counter+1
    ox=sorted(cx)
    a=0



    counter=0
    while(counter< (len(contour))):
        
            rect_list=cv2.boundingRect(contour[counter])
            if(ox[0]==rect_list[0]):
                os[0]=cv2.boundingRect(contour[counter])
                first=rect_list[1]-5
                second=rect_list[1]+rect_list[3]+5
                third=rect_list[0]-15
                fourth=rect_list[0]+rect_list[2]+5
                ss0=i1[first:second,third:fourth]
            counter=counter+1
    a=0

    counter=0
    while(counter< (len(contour))):
        
            rect_list=cv2.boundingRect(contour[counter])
            if(ox[1]==rect_list[0]):
                os[1]=cv2.boundingRect(contour[counter])
                first=rect_list[1]-5
                second=rect_list[1]+rect_list[3]+5
                third=rect_list[0]-15
                fourth=rect_list[0]+rect_list[2]+5
                ss1=i1[first:second,third:fourth]
        
            counter=counter+1
    a=0

    counter=0
    while(counter<(len(contour))):
        
            rect_list=cv2.boundingRect(contour[counter])
            if(ox[2]==rect_list[0]):
                os[2]=cv2.boundingRect(contour[counter])
                first=rect_list[1]-5
                second=rect_list[1]+rect_list[3]+5
                third=rect_list[0]-15
                fourth=rect_list[0]+rect_list[2]+5
                ss2=i1[first:second,third:fourth]
            counter=counter+1
    
    return ss0,ss1,ss2


#Reading the image
I= imread('im1.png')
i1 = resize(I, (960, 340))
hsv1 = RGB2HSV(i1)
v1 = RGB2GRAY(hsv1)
binary1 = binary_fun(v1)
er,de = d_e(i1) 

dilation=Pre(i1)
(s1,s2,s3)=Seg(i1)
plt.figure()
plt.subplot(3,3,1)
plt.imshow(i1)
plt.title("Input Sample")
plt.subplot(3,3,2)
plt.imshow(RGB2GRAY(i1),'gray')
plt.title("GRAY")
plt.subplot(3,3,4)
plt.imshow(de,'gray')
plt.title("Dilation")
plt.subplot(3,3,3)
plt.imshow(er,'gray')
plt.title("Erosion")
plt.subplot(3,3,5)
plt.imshow(s1,'gray')
plt.title("S1")
plt.subplot(3,3,6)
plt.imshow(s2,'gray')
plt.title("S2")
plt.subplot(3,3,7)
plt.imshow(s3,'gray')
plt.title("S3")



s1_cluster = cluster(s1)
s1_white = celldensity(s1)

s2_cluster = cluster(s2)
s2_white = celldensity(s2)

s3_cluster = cluster(s3)
s3_white = celldensity(s3)


if s1_cluster==1 and s1_white == 1:
    A = 1
    #print("Blood Group: A")
    ans="Blood Group: A"
else:
    A = 0
if s2_cluster==1 and s2_white == 1:
    B = 1
    #print("Blood Group: B")
    ans="Blood Group: B"
else:
    B = 0
if s3_cluster == 1 and s3_white == 1:
    m = 1
else:
    m = 0

if A != 1 and B!=1 :
    #print("Blood Group: 0")
    ans="Blood Group: 0"
if m == 1:
    #print(ans+" Positive")
    ans=ans +  " +ve"
else:
    #print(ans+" Negative")
    ans=ans +  " -ve"
print(ans)
plt.show()