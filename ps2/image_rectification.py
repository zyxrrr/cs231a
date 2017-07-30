import numpy as np
import matplotlib.pyplot as plt
from fundamental_matrix_estimation import *


'''
COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
given matching points in two images and the fundamental matrix
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the Fundamental matrix such that (points1)^T * F * points2 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(points1, points2, F):
    # TODO: Implement this method!
    epline2=F.dot(points2.T)
    A=epline2.T
    U,s,V=np.linalg.svd(A)
    ep1=V[0:1,:]
    epipole=ep1/ep1[0,2]
    return epipole
    raise Exception('Not Implemented Error')
    
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    # TODO: Implement this method!
    h,w=im2.shape[:2]
    T=np.array([[1,0,-w/2],[0,1,-h/2],[0,0,1]])
    e2T=T.dot(e2.T)
    e2T=e2T/e2T[2,0]
    if e2T[0,0]>=0:
        a=1
    else:
        a=-1 
    R=np.array([[a*e2T[0,0]/(np.sqrt(e2T[0,0]*e2T[0,0]+e2T[1,0]*e2T[1,0])),a*e2T[1,0]/(np.sqrt(e2T[0,0]*e2T[0,0]+e2T[1,0]*e2T[1,0])),0],\
                [a*e2T[1,0]/(np.sqrt(e2T[0,0]*e2T[0,0]+e2T[1,0]*e2T[1,0])),a*e2T[0,0]/(np.sqrt(e2T[0,0]*e2T[0,0]+e2T[1,0]*e2T[1,0])),0],\
                [0,0,1]])
    e2R=R.dot(e2T)
    G=np.array([[1,0,0],[0,1,0],[-1/e2R[0,0],0,1]])
    H2=np.linalg.inv(T).dot(G.dot(R.dot(T)))
    ex=np.array([[0,-e2[0][2],e2[0][1]],[e2[0][2],0,-e2[0][0]],[-e2[0][1],e2[0][0],0]])
    #return ex 
    M=ex.dot(F)+(e2.T).dot(np.array([[1,1,1]]))                                                     
    p2t=H2.dot(points2.T)
    p1t=(H2.dot(M)).dot(points1.T)
    p1temp=np.matlib.repmat(p1t[2:,:],3,1)
    p1t=np.divide(p1t,p1temp)
    p2temp=np.matlib.repmat(p2t[2:,:],3,1)                       
    p2t=np.divide(p2t,p2temp)
    W=p1t.T
    b=(p2t[0:1,:]).T
    a=np.linalg.lstsq(W,b)[0]
    Ha=np.array([[a[0,0],a[1,0],a[2,0]],[0,1,0],[0,0,1]])
    H1=Ha.dot(H2.dot(M))
    return H1,H2                           
    raise Exception('Not Implemented Error')

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print "F\n", F
    print "e1", e1
    print "e2", e2

    # Find the homographies needed to rectify the pair of images
    #H1 = compute_matching_homographies(e2, F, im2, points1, points2)
    #print "H1:\n", H1                           
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print "H1:\n", H1
    print
    print "H2:\n", H2

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
