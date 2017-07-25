"""CS231A Homework 1, Problem 2.

DATA FORMAT
In this problem, we provide and load the data for you. Recall that in the
original problem statement, there exists a grid of black squares on a white
background. We know how these black squares are setup, and thus can determine
the locations of specific points on the grid (namely the corners). We also have
images taken of the grid at a front image (where Z = 0) and a back image (where
Z = 150). The data we load for you consists of three parts: real_XY,
front_image, and back_image. For a corner (0,0), we may see it at the (137, 44)
pixel in the front image and the (148, 22) pixel in the back image. Thus, one
row of real_XY will contain the numpy array [0, 0], corresponding to the real
XY location (0, 0). The matching row in front_image will contain [137, 44] and
the matching row in back_image will contain [148, 22].
"""

import numpy as np
from scipy import linalg

def compute_camera_matrix(real_XY, front_image, back_image):
    """Computes camera matrix given image and real-world coordinates.

    Args:
        real_XY: Each row corresponds to an actual point on the 2D plane.
        front_image: Each row is the pixel location in the front image (Z=0).
        back_image: Each row is the pixel location in the back image (Z=150).
    Returns:
        camera_matrix: The calibrated camera matrix (3x4 matrix).
    """
    # TODO: Fill in this code.
    dims=real_XY.shape
    A_temp=np.ones((dims[0],1));
    A1=np.concatenate((real_XY,0*A_temp,A_temp),axis=1)
    b1=front_image[:,0]
    A2=np.concatenate((real_XY,150*A_temp,A_temp),axis=1)
    b2=back_image[:,0]
    A=np.concatenate((A1,A2),axis=0)
    b=np.concatenate((b1,b2),axis=0)
    affine1=np.linalg.lstsq(A,b)[0]
    b_y=np.concatenate((front_image[:,1],back_image[:,1]),axis=0)
    affine2=np.linalg.lstsq(A,b_y)[0]
    camera_matrix=np.concatenate((affine1.T,affine2.T,np.array([0,0,0,1])),axis=0)
    return camera_matrix
    pass


def rms_error(camera_matrix, real_XY, front_image, back_image):
    """Computes RMS error of points reprojected into the images.

    Args:
        camera_matrix: The camera matrix of the calibrated camera.
        real_XY: Each row corresponds to an actual point on the 2D plane.
        front_image: Each row is the pixel location in the front image (Z=0).
        back_image: Each row is the pixel location in the back image (Z=150).
    Returns:
        rms_error: The root mean square error of reprojecting the points back
            into the images.
    """
    # TODO: Fill in this code.
    dims=real_XY.shape
    A_temp=np.ones((dims[0],1));
    A1=np.concatenate((real_XY,0*A_temp,A_temp),axis=1)
    #b1=front_image[:,0]
    A2=np.concatenate((real_XY,150*A_temp,A_temp),axis=1)
    #b2=back_image[:,0]
    A=np.concatenate((A1,A2),axis=0)
    A=A.T
    camera_matrix=np.reshape(camera_matrix,(3,4))
    #print camera_matrix
    comCoor=camera_matrix.dot(A)
    comCoor=comCoor[:2,:]
    corr=np.concatenate((front_image,back_image),axis=0)
    corr=corr.T
    err=comCoor-corr
    err=np.multiply(err,err)
    err=np.sum(err,axis=0)
    err=np.sum(err/(err.shape[0]))
    return np.sqrt(err)
    pass

if __name__ == '__main__':
    # Load the example coordinates setup.
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)    
    rmse = rms_error(camera_matrix, real_XY, front_image, back_image)
    print "Camera Matrix:\n", camera_matrix
    print
    print "RMS Error: ", rmse

