# Author: Jesús Sánchez de Castro

import numpy as np

def pixel_lbp(center, neighbor):
    """
    For a given center pixel and a neighbor, check:
        - Neighbor >= center : 1
        - Neighbor < center : 0
        
    Parameters
    ----------
    center: pixel whose lbp value is being calculated
    neighbor: one of the sorrounding pixels of center.
    
    Returns
    -------
    [0,1] under the conditions exposed above.
    """
    value = 0
    
    if neighbor >= center:
        value = 1
    
    return value

def lbp_neighbors(img, x, y):
    """
    For a given pixel, calculate the value of the pixel's neighbors. 
    If the value of the neighbor is bigger or equal to the value of the pixel, 
    1 is returned, if the nieghbor is smaller, 0 is returned. This value for a 
    pair of pixels (center, neighbor) is calculated in pixel_lbp().
    
    130 | 130 | 120
    - - - - - - - -
    130 | 120 | 110    -> | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 
    - - - - - - - -                      *                  = sum(product)=231
    130 | 120 | 110       [ 1 , 2 , 4 , 8 , 16, 32, 62,128]
    
    l = [[130,130,120],[130,120,110],[130,120,110]]
    Parameters
    ----------

    img: Image.
    x, y: coordinates of the center pixel.
    Returns
    -------
    
    lbp value of the given pixel.
    """
    center_pixel = img[x][y]
    
    neighbors_value = []
    neighbors_value.append(pixel_lbp(center_pixel, img[x+1][y-1])) # TOP LEFT
    neighbors_value.append(pixel_lbp(center_pixel, img[x-1][y]))   # TOP
    neighbors_value.append(pixel_lbp(center_pixel, img[x-1][y+1])) # TOP RIGHT
    neighbors_value.append(pixel_lbp(center_pixel, img[x][y+1]))   # RIGHT
    neighbors_value.append(pixel_lbp(center_pixel, img[x+1][y+1])) # BOTTOM RIGHT
    neighbors_value.append(pixel_lbp(center_pixel, img[x+1][y]))   # BOTTOM
    neighbors_value.append(pixel_lbp(center_pixel, img[x+1][y-1])) # BOTTOM LEFT
    neighbors_value.append(pixel_lbp(center_pixel, img[x][y-1]))   # LEFT
    
#    print(neighbors_value)
    binary_values = [1,2,4,8,16,32,64,128]
    value = 0
    
    for i in range(0, len(neighbors_value)):
#        print(neighbors_value[i])
#        print(neighbors_value[i]*binary_values[i])
        value += neighbors_value[i]*binary_values[i]
    return value
    
def lbp_compute(img):
    
    # Matrix where the lbp values are going to be put
    img_lbp = np.zeros(shape=(img.shape[0], img.shape[1])).astype(int)
        
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            img_lbp[i][j] = lbp_neighbors(img, i, j)
            
    return img_lbp

def lbp_hist(lbp_img, step, win_size, uniform):

    width = lbp_img.shape[0]
    height = lbp_img.shape[1]
    check_uniform = [0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,
               58,12,58,58,58,13,58,14,15,16,58,58,58,58,58,58,58,58,58,58,
               58,58,58,58,58,17,58,58,58,58,58,58,58,18,58,58,58,19,58,20,
               21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
               58,58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,
               58,58,58,58,58,58,58,58,58,24,58,58,58,58,58,58,58,25,58,58,
               58,26,58,27,28,29,30,58,31,58,58,58,32,58,58,58,58,58,58,58,
               33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,
               58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
               58,58,58,58,58,58,58,58,35,36,37,58,38,58,58,58,39,58,58,58,
               58,58,58,58,40,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
               41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,
               49,58,58,58,50,51,52,58,53,54,55,56,57]
    descriptor = []
    
    for j in range(0, height-(win_size-1), step):
        for i in range(0, width-(win_size-1), step):
#            print("i:",i,i+win_size,"j:",j,j+win_size)
            window = lbp_img[i:i+win_size,j:j+win_size]
            window = window.flatten()
            
            if uniform: 
                hist = [0]*59
                for pixel in range(0, len(window)):
                    pixel_value = window[pixel]
                    hist[check_uniform[pixel_value]] += 1
                descriptor.append(hist)
            else: 
                hist = [0]*256
                for pixel in range(0, len(window)):
                    pixel_value = window[pixel]
                    hist[pixel_value] += 1
                descriptor.append(hist)
                
            
    return descriptor