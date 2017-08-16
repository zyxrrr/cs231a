import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math


'''
SELECT_KEYPOINTS_IN_BBOX: Return only the keypoints and corresponding 
descriptors whose pixel locations are within a specified bounding box.

Arguments:
    descriptors - Descriptors of the image. Each row corresponds to a 
        SIFT descriptor of a keypoint. This is a ndarray of size (M, 128).

    keypoints - Each row is a SIFT keypoint consisting of (u, v, scale, 
        theta). Overall, this variable is a ndarray of size (M, 4).

Returns:
    descriptors - The descriptors corresponding to the keypoints inside the
        bounding box.

    keypoints - The pixel locations of the keypoints that reside in the 
        bounding box
'''
def select_keypoints_in_bbox(descriptors, keypoints, bbox):
    xmin, ymin, xmax, ymax = bbox
    indices = [i for i, pt in enumerate(keypoints) if 
            pt[0] >= xmin and pt[0] <= xmax and pt[1] >= ymin and pt[1] <= ymax]
    return descriptors[indices, :], keypoints[indices, :]


'''
PLOT_MATCHES: Given two images, plots the matching points between them

Arguments:
    im1 - The first image.
    
    im2 - The second image

    p1 - The keypoints of the first image.

    p2 - The keypoints of the second image

    matches - An int ndarray of size (N, 2) of indices that for keypoints in 
        descriptors1 match which keypoints in descriptors2. For example, [7 5]
        would mean that the keypoint at index 7 of descriptors1 matches the
        keypoint at index 5 of descriptors2. Not every keypoint will necessarily
        have a match, so N is not the same as the number of rows in descriptors1
        or descriptors2. 
Returns:
    Nothing; instead generates a figure in which it draws a line between the
        matching points of image 1 and image 2.
'''
def plot_matches(im1, im2, p1, p2, matches):
    fig = plt.figure()
    new_im = np.zeros((max(im1.shape[0], im2.shape[0]), im1.shape[1]+im2.shape[1], max(im1.shape[2], im2.shape[2])),dtype=np.uint8)
    new_im[:im1.shape[0], :im1.shape[1], :im1.shape[2]] = im1
    new_im[:im2.shape[0], im1.shape[1]:, :im2.shape[2]] = im2
    plt.imshow(new_im)
    plt.autoscale(False)
    for m in matches:
        ind1, ind2 = m
        plt.plot([p1[ind1,0], im1.shape[1]+p2[ind2,0]], [p1[ind1,1], p2[ind2,1]], '-x')
    plt.show()


'''
PLOT_BBOX: Given an image, plots the detected regions as a bounding box.

Arguments:
    cx - A vector of the x coodinate of the center of the bounding boxes.
    
    cy - A vector of the y coodinate of the center of the bounding boxes.
    
    w - A vector of the width of the bounding boxes.
    
    h - A vector of the height of the bounding boxes.

    orient - A vector of the orientation of the bounding boxes.
    
    im - The image.

Returns:
    Nothing; instead generates a figure in which all the detected bounding
        boxes are drawn on top of the image
'''
def plot_bbox(cx, cy, w, h, orient, im):
    N = len(cx)
    plt.figure()
    plt.imshow(im)
    for k in xrange(N):
        x = cx[k] + np.hstack([-w[k]/2*math.cos(orient[k])-h[k]/2*math.sin(orient[k]),
            -w[k]/2*math.cos(orient[k])+h[k]/2*math.sin(orient[k]), 
            w[k]/2*math.cos(orient[k])+h[k]/2*math.sin(orient[k]),
            w[k]/2*math.cos(orient[k])-h[k]/2*math.sin(orient[k])])
        x = np.hstack((x, x[0]))
        y = cy[k] + np.hstack([w[k]/2*math.sin(orient[k])-h[k]/2*math.cos(orient[k]),
            w[k]/2*math.sin(orient[k])+h[k]/2*math.cos(orient[k]),
            -w[k]/2*math.sin(orient[k])+h[k]/2*math.cos(orient[k]),
            -w[k]/2*math.sin(orient[k])-h[k]/2*math.cos(orient[k])])
        y = np.hstack((y, y[0]))
        plt.plot(x, y, c='g', linewidth=5);
        plt.plot(x, y, c='k', linewidth=1);         
    plt.show()
