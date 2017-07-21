# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
#from scipy.misc import imread, imsave, imresize


def main():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE    
    img1=misc.imread('./image1.jpg') 
    img2=misc.imread('./image2.jpg')   
    
    # END YOUR CODE HERE

    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    img1.astype(float)
    img2.astype(float)
    img1nom=np.multiply(img1,1.0/255)
    img2nom=np.multiply(img2,1.0/255)
    # END YOUR CODE HERE

    # ===== Problem 3c =====
    # Add the images together and re-normalize them 
    # to have minimum value 0 and maximum value 1. 
    # Display this image.

    # BEGIN YOUR CODE HERE
    imgAdd=img1+img2
    imgAddNom=np.multiply(imgAdd,0.5/255)
    #plt.subplot(2,1,1)
    plt.figure(1);
    plt.imshow(imgAddNom)
    
    # END YOUR CODE HERE

    # ===== Problem 3d =====
    # Create a new image such that the left half of 
    # the image is the left half of image1 and the 
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE
    imgSize=img1.shape
    mid=imgSize[1]/2;
    #newImage1=np.zeros(imgSize)
    #newImage1[:,:mid,:]=img1[:,:mid,:]
    #newImage1[:,mid:,:]=img2[:,mid:,:]
    #plt.subplot(2,1,2)

    img1_left=img1[:,:mid]
    img2_right=img2[:,mid:]
    newImage1=np.concatenate((img1_left ,img2_right),axis=1)
    
    plt.figure(2)
    plt.imshow(newImage1)
    
    # END YOUR CODE HERE

    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd 
    # numbered row is the corresponding row from image1 and the 
    # every even row is the corresponding row from image2. 
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = None
    newImage2=np.zeros(imgSize)

    # BEGIN YOUR CODE HERE
    for idx in range(300):
        if idx%2==0:
            newImage2[idx,:,:]=img1[idx,:,:]
        else:
            newImage2[idx,:,:]=img2[idx,:,:]
    plt.figure(3)
    plt.imshow(newImage2)

    
                

    # END YOUR CODE HERE

    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and repmat may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE
    newIma = np.concatenate((img1, img2), axis=1)
    newIma = np.reshape(newIma, (150, 1200, 3))
    #print('jj', newImage2.shape)
    newIma = newIma[:, :600]
    newImage3 = np.reshape(newIma, (300, 300, 3))
    plt.figure(4)
    plt.imshow(newImage3)

    

    # END YOUR CODE HERE

    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image. 
    # Display the grayscale image with a title.

    # BEGIN YOUR CODE HERE
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    grayimg=rgb2gray(newImage3)
    fig3=plt.figure(num= 5)
    fig3.suptitle('grey image',fontsize=20)

    plt.imshow(grayimg,cmap=plt.get_cmap('gray'))
    plt.show()
    # END YOUR CODE HERE
    plt.close()


    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
