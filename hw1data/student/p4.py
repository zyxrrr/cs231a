# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import linalg

def main():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.

    img1 = None

    # BEGIN YOUR CODE HERE
    img1=misc.imread('./image1.jpg',mode='F')
    plt.figure(1)
    plt.imshow(img1,cmap='gray')
    U,s,Vh=linalg.svd(img1)

    # END YOUR CODE HERE

    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.

    rank1approx = None

    # BEGIN YOUR CODE HERE
    tmp=np.dot(U[:, 0], s[0])
    tmp=np.reshape(tmp,(300,1))
    #print('tmp',tmp.shape)
    a=np.reshape(Vh[0,:],(1,300))
    rank1approx=np.dot(tmp,a)
    #print(rank1approx.shape)
    plt.figure(2)
    plt.imshow(rank1approx)

    # END YOUR CODE HERE

    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    rank20approx = None

    # BEGIN YOUR CODE HERE
    s=np.diag(s[0:20])
    #print('s',s)
    tmp = np.dot(U[:, 0:20], s)
    #print('tmp', tmp.shape)
    tmp = np.reshape(tmp, (300, 20))
    #print('tmp', tmp.shape)
    a = np.reshape(Vh[0:20, :], (20, 300))
    rank20approx = np.dot(tmp, a)
    #print(rank20approx.shape)
    plt.figure(3)
    plt.imshow(rank20approx,cmap='gray')
    plt.show()
    plt.close()
    misc.imsave('rank20approx_image1.png',rank20approx)

    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
