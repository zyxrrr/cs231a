import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    # TODO: Implement this method!
    points1Ave=np.average(points_im1,axis=0)
    points2Ave=np.average(points_im2,axis=0)
    pointsSize=points_im1.shape
    points1temp=np.matlib.repmat(points1Ave,pointsSize[0],1)
    points2temp=np.matlib.repmat(points2Ave,pointsSize[0],1)
    p1Centered=points_im1-points1temp
    p2Centered=points_im2-points2temp
    p1Centered=p1Centered[:,:2]
    p2Centered=p2Centered[:,:2]
    D=np.concatenate((p1Centered.T,p2Centered.T),axis=0)
    U,s,V=np.linalg.svd(D)
    print 's:\n',s
    S=np.diag(s)[:3,:3]
    motion=U[:,:3].dot(np.sqrt(S))    
    structure=(np.sqrt(S)).dot(V[:3,:])
    return structure,motion
    raise Exception('Not Implemented Error')

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        #print 'points1:\n',points_im1
        #print 'points2:\n',points_im2
        #print 'points_3d\n',points_3d
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()
