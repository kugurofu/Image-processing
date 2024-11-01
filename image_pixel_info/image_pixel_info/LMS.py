import numpy as np
import skimage
import cv2

def srgb_to_lrgb(srgb): 
    return skimage.exposure.adjust_gamma(srgb,2.2)
    
def lrgb_to_srgb(linear): 
    return skimage.exposure.adjust_gamma(np.clip(linear,0.0,1.0),1/2.2)
    
RGB_TO_XYZ=np.array(
    [[0.41245, 0.35758, 0.18042],
    [0.21267, 0.71516, 0.07217],
    [0.01933, 0.11919, 0.95023]])
    
XYZ_TO_RGB=np.array(
    [[3.24048, -1.53715, -0.49854],
    [-0.96926, 0.71516, 0.07217],
    [0.01933, 0.11919, 0.95023]])
    
def srgb_to_xyz(srgb): 
    return srgb_to_lrgb(srgb) @ RGB_TO_XYZ.T    

def xyz_to_srgb(xyz): 
    return lrgb_to_srgb(xyz@XYZ_TO_RGB.T)
    
def xyz_to_lms(xyz,M): 
    return xyz@M.T

def normalize_xyz(xyz): 
    return xyz/xyz[1]
    
def ave_srgb(img): 
    return lrgb_to_srgb(srgb_to_lrgb(img).mean((0,1)))
    
def chromatic_adaptation(src_white_point,dst_white_point,src_img,adapt): 
    src_img_xyz=srgb_to_xyz(src_img)
    xyz_src=normalize_xyz(srgb_to_xyz(src_white_point))
    xyz_dst=normalize_xyz(srgb_to_xyz(dst_white_point))
    XYZ_TO_LMS=np.array(
    [[0.733, 0.430, -0.162],
    [-0.704, 1.698, 0.006],
    [0.003, 0.014, 0.983]])
    lms_src = xyz_to_lms(xyz_src,XYZ_TO_LMS)
    lms_dst = xyz_to_lms(xyz_dst,XYZ_TO_LMS)
    g=(adapt*lms_dst+(1.0-adapt)*lms_src)/lms_src
    adapt_mat=np.linalg.inv(XYZ_TO_LMS)@np.diag(g)@XYZ_TO_LMS
    adapt_xyz=src_img_xyz@adapt_mat.T
    return xyz_to_srgb(adapt_xyz)

img=cv2.imread('original.jpg')
img=img[:,:,::-1]/255
dst_white_point=np.array([1.,1.,1.])

src_white_point=ave_srgb(img)
adapt_img=chromatic_adaptation(src_white_point,dst_white_point,img,1.0)
