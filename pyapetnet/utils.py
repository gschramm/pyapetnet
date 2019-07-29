import os
from datetime import datetime
import string

import numpy as np
#----------------------------------------------------------------------------------------------

def cross_product_matrix(a):
  # cross product matrix of a vector a
  # useful to contruct an affine matrix for a
 
  ndim = a.shape[0]
  cpm  = np.zeros((ndim,ndim))

  for i in range(ndim):
    uv       = np.zeros(ndim)
    uv[i]    = 1
    cpm[:,i] = np.cross(a, uv) 

  return cpm

#----------------------------------------------------------------------------------------------

def rotation_matrix(uv, theta):
  # general rotation matrix for rotation around unit vector uv

  ndim = uv.shape[0]

  tmp = np.cos(theta)*np.eye(ndim) + np.sin(theta)*cross_product_matrix(uv) + (1 - np.cos(theta))*np.outer(uv,uv)
 
  R = np.zeros((ndim + 1, ndim + 1))
  R[:-1,:-1] = tmp
  R[-1,-1]   = 1

  return R

#----------------------------------------------------------------------------------------------

def affine_center_rotation(uv, theta, uv_origin = None, offset = None):
  # affine trasnformation for rotation around unit vector uv through the center followd by shift

  ndim = uv.shape[0]

  # set up affine to transform origin of rotation axis into image center
  T = np.eye(ndim + 1, ndim + 1)
  if uv_origin is not None: T[:-1,-1] -= uv_origin 

  R  = rotation_matrix(uv, theta)

  aff = np.linalg.inv(T) @ (R @ T)

  if offset is not None: aff[:-1,-1] += offset

  return aff

#----------------------------------------------------------------------------------------------

def time_str():
    timestamp = str(datetime.now())
    s = ''
    for c in timestamp:
        if c not in string.ascii_letters:
            if c not in string.digits:
                s += '_'
                continue
        s += c

    return s

def ispath(p):
    if type(p) is str:
        if os.path.exists(p):
            return True
    
    return False

def normalize(x0):
    intercept = x0.min()
    slope = x0.max() - x0.min()
    x = (x0 - intercept)/slope

    norm_fun = lambda x : (x-intercept)/slope
    norm_inv = lambda x : x*slope + intercept

    return x, norm_fun, norm_inv

def yell(msg, content, yell_char='!'):
    
    
    line2 = '!!  {}  !!'.format(msg)
    line1 = '!'*len(line2)
    
    header = '\n'.join((line1, line2, line1))
    header = header.replace('!', yell_char)
    footer = line1.replace('!',yell_char)

    val = '\n'.join((header, content, footer))

    return val

#----------------------------------------------------------------------------------------------

if __name__ == '__main__':

  import numpy as np
  import pynucmed as ni

  from scipy.ndimage.interpolation import shift, rotate, affine_transform

  n  = 97
  m  = int(n//4)
  img = np.zeros((n,n,n))
  img[m:-m,m:-m,m:-m] = 1

  img_center = np.array(img.shape)/2 - 0.5

  uv = np.array([1,0,0])
  uv = uv / np.linalg.norm(uv) 

  aff1 = affine_center_rotation(uv, 45*np.pi/180, uv_origin = 0.5*img_center)
  img1 = affine_transform(img, aff1, order = 1, prefilter = False)

  vi = ni.viewer.ThreeAxisViewer([img,img1])
   

    
        
        
    
