import numpy as np
import skimage.io as sio
from scipy.io import loadmat
from plotting import *
import math


'''
COMPUTE_GRADIENT Given an image, computes the pixel gradients

Arguments:
    im - a grayscale image, represented as an ndarray of size (H, W) containing
        the pixel values

Returns:
    angles - ndarray of size (H-2, W-2) containing gradient angles in degrees

    magnitudes - ndarray of size (H-2, W-2) containing gradient magnitudes

The way that the angles and magnitude per pixel are computed as follows:
Given the following pixel grid

    P1 P2 P3
    P4 P5 P6
    P7 P8 P9

We compute the angle on P5 as arctan(dy/dx) = arctan(P2-P8 / P4-P6).
Note that we should be using np.arctan2, which is more numerically stable.
However, this puts us in the range [-180, 180] degrees. To be in the range
[0,180], we need to simply add 180 degrees to the negative angles.

The magnitude is simply sqrt((P4-P6)^2 + (P2-P8)^2)
'''
def compute_gradient(im):
    # TODO: Implement this method!
    #print 'im\n',im
    im_size=im.shape
    angles=np.zeros((im_size[0]-2,im_size[1]-2))
    magnitudes=np.zeros((im_size[0]-2,im_size[1]-2))
    for hh in range(1,im_size[0]-1):
        for ww in range(1,im_size[1]-1):
            patch=im[hh-1:hh+2,ww-1:ww+2]
            temp1=(patch[0,1]-patch[2,1])/(patch[1,0]-patch[1,2])
            #print temp1
            angle=np.arctan2(patch[0,1]-patch[2,1],patch[1,0]-patch[1,2])
            angle*= 180 / np.pi
            if angle<0:
                angle=angle+180
            temp2=(patch[1,0]-patch[1,2])**2+(patch[0,1]-patch[2,1])**2
            mag=np.sqrt(temp2)
            angles[hh-1,ww-1]=angle
            magnitudes[hh-1,ww-1]=mag
    return angles,magnitudes
    raise Exception('Not Implemented Error')
    

'''
GENERATE_HISTOGRAM Given matrices of angles and magnitudes of the image
gradient, generate the histogram of angles

Arguments:
    angles - ndarray of size (M, N) containing gradient angles in degrees

    magnitudes - ndarray of size (M, N) containing gradient magnitudes

    nbins - the number of bins that you want to bin the histogram into

Returns:
    histogram - an ndarray of size (nbins,) containing the distribution
        of gradient angles.

This method should be implemented as follows:

1)Each histogram will bin the angle from 0 to 180 degrees. The number of bins
will dictate what angles fall into what bin (i.e. if nbins=9, then first bin
will contain the votes of angles close to 10, the second bin will contain those
close to 30, etc).

2) To create these histogram, iterate over the gradient grids, putting each
gradient into its respective bins. To do this properly, we interpolate and
weight the voting by both its magnitude and how close it is to the average
angle of the two bins closest to the angle of the gradient. For example, if we
have nbins = 9 and receive angle of 20 degrees with magnitude 1, then we the
vote contribution to the histogram weights equally with the first and second bins
(since its closest to both 10 and 30 degrees). If instead, we recieve angle of
25 degrees with magnitude 2, then it is weighted 25% in the first bin and 75%
in second bin, but with twice the voting power.

Mathematically, if you have an angle, magnitude, the center_angle1 of the lower
bin center_angle2 of the higher bin, then:

histogram[bin1] += magnitude * |angle - center_angle2| / (180 / nbins) 
histogram[bin2] += magnitude * |angle - center_angle1| / (180 / nbins)

Notice how that we're weighting by the distance to the opposite center. The
further the angle is from one center means it is closer to the opposite center
(and should be weighted more).

One special case you will have to take care of is when the angle is near
180 degrees or 0 degrees. It may be that the two nearest bins are the first and
last bins respectively.OA
'''
def generate_histogram(angles, magnitudes, nbins = 9):
    # TODO: Implement this method!
    #print "angles:\n",angles
    #print "mags:\n",magnitudes
    histogram=np.zeros((nbins,1))
    center_angles=np.arange(nbins)
    step=180/nbins
    center_angles=step*center_angles+10
    #print "center angle:\n",center_angles
    for ii in range(nbins-1):
        idx=np.logical_and(angles>=center_angles[ii],angles<center_angles[ii+1])
        #print "idx:\n",idx
        temp=magnitudes[idx]
        #print "temp:\n",temp
        a=(angles[idx]-center_angles[ii+1])
        temp2=np.abs(a/(180.0 / nbins))#notice int divide int is int!!!
        histogram[ii]+=np.sum(np.multiply(temp,temp2))
        a=(angles[idx]-center_angles[ii])
        temp2=np.abs(a/(180.0 / nbins))
        histogram[ii+1]+=np.sum(np.multiply(temp,temp2))

    idx=angles>=center_angles[nbins-1]
    temp=magnitudes[idx]
    histogram[0]+=np.sum(np.multiply(temp,np.abs((angles[idx]-center_angles[nbins-1])/(180.0 / nbins))))
    histogram[nbins-1]+=np.sum(np.multiply(temp,np.abs((180-angles[idx]+center_angles[0])/(180.0 / nbins))))

    idx=angles<center_angles[0]
    temp=magnitudes[idx]
    histogram[0]+=np.sum(np.multiply(temp,np.abs((angles[idx]+180-center_angles[nbins-1])/(180.0 / nbins))))
    histogram[nbins-1]+=np.sum(np.multiply(temp,np.abs((angles[idx]-center_angles[0])/(180.0 / nbins))))
    #print histogram
    return histogram.flatten()
    raise Exception('Not Implemented Error')


'''
COMPUTE_HOG_FEATURES Computes the histogram of gradients features
Arguments:
    im - the image matrix

    pixels_in_cell - each cell will be of size (pixels_in_cell, pixels_in_cell)
        pixels

    cells_in_block - each block will be of size (cells_in_block, cells_in_block)
        cells

    nbins - number of histogram bins

Returns:
    features - the hog features of the image represented as an ndarray of size
        (H_blocks, W_blocks, cells_in_block * cells_in_block * nbins), where
            H_blocks is the number of blocks that fit height-wise
            W_blocks is the number of blocks that fit width-wise

Generating the HoG features can be done as follows:

1) Compute the gradient for the image, generating angles and magnitudes

2) Define a cell, which is a grid of (pixels_in_cell, pixels_in_cell) pixels.
Also, define a block, which is a grid of (cells_in_block, cells_in_block) cells.
This means each block is a grid of side length pixels_in_cell * cell_in_block
pixels.

3) Pass a sliding window over the image, with the window size being the size of
a block. The stride of the sliding window should be half the block size, (50%
overlap). Each cell in each block will store a histogram of the gradients in
that cell. Consequently, there will be cells_in_block * cells_in_block
histograms in each block. This means that each block feature will initially
represented as a (cells_in_block, cells_in_block, nbins) ndarray, that can
reshaped into a (cells_in_block * cells_in_block *nbins,) ndarray. Make sure to
normalize such that the norm of this flattened block feature is 1. 

4) The overall hog feature that you return will be a grid of all these flattened
block features.

Note: The final overall feature ndarray can be flattened if you want to use to
train a classifier or use it as a feature vector.
'''
def compute_hog_features(im, pixels_in_cell, cells_in_block, nbins):
    # TODO: Implement this method!
    #print "im:\n",im
    #print "pixels in cell:\n",pixels_in_cell
    #print "cell in block:\n",cells_in_block
    im_size=im.shape
    block_size=pixels_in_cell*cells_in_block
    step=pixels_in_cell*cells_in_block/2
    hh=range(0,im_size[0]-pixels_in_cell*cells_in_block+1,step)
    ww=range(0,im_size[1]-pixels_in_cell*cells_in_block+1,step)
    H_blocks=len(hh)
    W_blocks=len(ww)
    temp=np.zeros((nbins,))
    features=np.zeros((H_blocks, W_blocks, cells_in_block * cells_in_block * nbins))
    for h in hh:
        for w in ww:
            patch=im[h:h+block_size,w:w+block_size]
            histogram_block=np.zeros((nbins,))
            for h_cell in range(cells_in_block):
                for w_cell in range(cells_in_block):
                    patch_cell=patch[h_cell*pixels_in_cell:(h_cell+1)*pixels_in_cell,w_cell*pixels_in_cell:(w_cell+1)*pixels_in_cell]
                    angles,magnitudes=compute_gradient(patch_cell)
                    histogram=generate_histogram(angles, magnitudes, nbins)
                    histogram_block=np.concatenate((histogram_block,histogram))
            histogram_block=histogram_block[nbins:]
            his_norn=np.linalg.norm(histogram_block)
            histogram_block=histogram_block/his_norn
            features[h/step,w/step,:]=histogram_block
    return features
    raise Exception('Not Implemented Error')


if __name__ == '__main__':
    # Part A: Checking the image gradient
    print '-' * 80
    print 'Part A: Image gradient'
    print '-' * 80
    im = sio.imread('simple.jpg', True)
    grad_angle, grad_magnitude = compute_gradient(im)
    print "Expected angle: 126.339396329"
    print "Expected magnitude: 0.423547566786"
    print "Checking gradient test case 1:", \
        np.abs(grad_angle[0][0] - 126.339396329) < 1e-3 and \
        np.abs(grad_magnitude[0][0] - 0.423547566786) < 1e-3

    im = np.array([[1, 2, 2, 4, 8],
                    [3, 0, 1, 5, 10],
                    [10, 13, 12, 2, 7],
                    [10, 5, 1, 0, 3],
                    [1, 1, 1.5, 2, 2.5]])
    grad_angle, grad_magnitude = compute_gradient(im)
    correct_angle = np.array([[ 100.30484647,   63.43494882,  167.47119229],
                              [  68.19859051,    0.        ,   45.        ],
                              [  53.13010235,   64.53665494,  180.        ]])
    correct_magnitude = np.array([[ 11.18033989,  11.18033989,   9.21954446],
                                  [  5.38516481,  11.        ,   7.07106781],
                                  [ 15.        ,  11.62970335,   2.        ]])
    print "Expected angles: \n", correct_angle
    print "Expected magnitudes: \n", correct_magnitude
    print "Checking gradient test case 2:", \
        np.allclose(grad_angle, correct_angle) and \
        np.allclose(grad_magnitude, correct_magnitude)

    # Part B: Checking the histogram generation
    print '-' * 80
    print 'Part B: Histogram generation'
    print '-' * 80
    angles = np.array([[10, 30, 50], [70, 90, 110], [130, 150, 170]])
    magnitudes = np.arange(1,10).reshape((3,3))
    print "Checking histogram test case 1:", \
        np.all(generate_histogram(angles, magnitudes, nbins = 9) == np.arange(1,10))

    angles = np.array([[20, 40, 60], [80, 100, 120], [140, 160, 180]])
    magnitudes = np.arange(1,19,2).reshape((3,3))
    histogram = generate_histogram(angles, magnitudes, nbins = 9)
    print "Checking histogram test case 2:", \
        np.all(histogram  == np.array([9, 2, 4, 6, 8, 10, 12, 14, 16]))

    angles = np.array([[13, 23, 14.3], [53, 108, 1], [77, 8, 32]])
    magnitudes = np.ones((3,3)) 
    histogram = generate_histogram(angles, magnitudes, nbins = 9)
    print "Submit these results:", histogram

    # Part C: Computing and displaying the final HoG features
    # vary cell size to change the output feature vector. These parameters are common parameters
    pixels_in_cell = 8
    cells_in_block = 2
    nbins = 9
    im = sio.imread('car.jpg', True)
    car_hog_feat = compute_hog_features(im, pixels_in_cell, cells_in_block, nbins)
    show_hog(im, car_hog_feat, figsize = (18,6))
