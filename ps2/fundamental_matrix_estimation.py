import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    temp1=np.matlib.repmat(points1[:,0:1],1,3)
    temp2=np.matlib.repmat(points1[:,1:2],1,3)
    temp3=np.matlib.repmat(points1[:,2:3],1,3)
    points1=np.concatenate((temp1,temp2,temp3),axis=1)
    temp=np.matlib.repmat(points2,1,3)
    W=np.multiply(points1,temp)
    U,s,V=np.linalg.svd(W)
    v_shape=V.shape
    f=V[v_shape[0]-1:v_shape[0],:]
    F=np.reshape(f,(3,3))
    U,s,V=np.linalg.svd(F)
    S=np.diag(s)
    S[2,2]=0
    #return S
    F=U.dot(S.dot(V))
    return F
    #temp1=np.reshape(temp1,
    raise Exception('Not Implemented Error')

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    p1Size=points1.shape
    #p1Sum=np.sum(points1,axis=0)/p1Size[0]
    p1Ave=np.average(points1,axis=0)
    Tra=np.array([[1,0,-p1Ave[0]],[0,1,-p1Ave[1]],[0,0,1]])
    p1s=points1.T
    p1Tra=Tra.dot(p1s)
    s=1/p1Tra.max()
    S=np.array([[s,0,0],[0,s,0],[0,0,1]])
    T1=S.dot(Tra)
    P1t=(T1.dot(p1s)).T
                  
    p2Size=points2.shape
    #p1Sum=np.sum(points1,axis=0)/p1Size[0]
    p2Ave=np.average(points2,axis=0)
    Tra=np.array([[1,0,-p2Ave[0]],[0,1,-p2Ave[1]],[0,0,1]])
    p2s=points2.T
    p2Tra=Tra.dot(p2s)
    s=1/p2Tra.max()
    S=np.array([[s,0,0],[0,s,0],[0,0,1]])
    T2=S.dot(Tra)
    P2t=(T2.dot(p2s)).T
    F_lls = lls_eight_point_alg(P1t, P2t)
    F=(T1.T).dot(F_lls.dot(T2))                 
    
    return F
    raise Exception('Not Implemented Error')

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    # TODO: Implement this method!
    pSize=points1.shape
    n=pSize[0]
    #n=5
    #print 'points size :\n',pSize
    plt.figure()
    plt.imshow(im1, cmap ='gray')
    for i in range(n):
        plot_epipolar_line(im1,F,points2[i:i+1,:])       
        plt.plot(points1[i,0],points1[i,1],'o')
    #plt.axis('off')
    

    plt.figure()
    plt.imshow(im2, cmap ='gray')
    for i in range(n):
        #plot_epipolar_line(im2,F,points1)
        plot_epipolar_line(im2,F.T,points1[i:i+1,:])
        plt.plot(points2[i,0],points2[i,1],'o')
    #plt.axis('off')

    #plt.show()
    #raise Exception('Not Implemented Error')



def plot_epipolar_line(img,F,points):
    m,n=img.shape[:2]
    line=F.dot(points.T)

    t=np.linspace(0,n,100)
    lt=np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

    ndx=(lt>=0)&(lt<m)
    t=np.reshape(t,(100,1))
    #print 't\n',t[ndx]
    #print 'lt\n',lt[ndx]
    plt.plot(t[ndx],lt[ndx])
    #plt.show()
    #raise Exception('plot_epipolar_line wrong!')

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # TODO: Implement this method!
    p2=points2.T
    epLine2=F.dot(p2)
    p1=points1.T
    temp1=np.sum(np.multiply(epLine2,p1),axis=0)
    epLine2[2:3,:]=0
    temp2=np.sqrt(np.sum(np.multiply(epLine2,epLine2),axis=0))
    dis1=np.divide(temp1,temp2)
    d1Ave=np.average(dis1)
    return d1Ave
    raise Exception('Not Implemented Error')

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print '-'*80
        print "Set:", im_set
        print '-'*80

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        #print 'points1\n',points1[0]
        #print 'size\n',points1.shape
        F_lls = lls_eight_point_alg(points1, points2)
        print "Fundamental Matrix from LLS  8-point algorithm:\n", F_lls
        print "Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls)
        print "Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T)

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)
        #print points2[1].dot(F_normalized.dot(points1[1]))

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in xrange(points1.shape[0])]
        print "p'^T F p =", np.abs(pFp).max()
        print "Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized
        print "Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized)
        print "Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
        plt.close()
