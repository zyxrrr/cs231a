import numpy as np
import os
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from utils import *

'''
RUN_DETECTOR Given an image, runs the SVM detector and outputs bounding
boxes and scores

Arguments:
    im - the image matrix

    clf - the sklearn SVM object. You will probably use the 
        decision_function() method to determine whether the object is 
        a face or not.
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    window_size - an array which contains the height and width of the sliding
    	window

    cell_size - each cell will be of size (cell_size, cell_size) pixels

    block_size - each block will be of size (block_size, block_size) cells

    nbins - number of histogram bins

Returns:
    bboxes - D x 4 bounding boxes that tell [xmin ymin width height] per bounding
    	box

    scores - the SVM scores associated with each bounding box in bboxes

You can compute the HoG features using the compute_hog_features() method
that you implemented in PS3. We have provided an implementation in utils.py,
but feel free to use your own implementation. You will use the HoG features
in a sliding window based detection approach.

Recall that using a sliding window is to take a certain section (called the 
window) of the image and compute a score for it. This window then "slides"
across the image, shifting by either n pixels up or down (where n is called 
the window's stride). 

Using a sliding window approach (with stride of block_size * cell_size / 2),
compute the SVM score for that window. If it's greater than 1 (the SVM decision
boundary), add it to the bounding box list. At the very end, after implementing 
nonmaximal suppression, you will filter the nonmaximal bounding boxes out.
'''
def run_detector(im, clf, window_size, cell_size, block_size, nbins, thresh=1):
    # TODO: Implement this method!
    step=block_size*cell_size/2
    im_size=im.shape
    bboxes=np.zeros((1,4))
    scores=np.zeros((1,1))
    for h in xrange(0,im_size[0]-window_size[0],step):
        for w in xrange(0,im_size[1]-window_size[1],step):
            #print "h w\n",h,w
            patch=im[h:h+window_size[0],w:w+window_size[1]]
            feature=compute_hog_features(patch, cell_size, block_size, nbins)
            #print feature.reshape(1,-1)
            face_decision=clf.decision_function(feature.reshape(1,-1))
            #print "sco:\n",clf.decision_function()
            #print "face:\n",face_decision
            if face_decision>thresh:
                bboxe=np.array([[w,h,window_size[1],window_size[0]]])
                bboxes=np.concatenate((bboxes,bboxe),axis=0)
                temp=np.array([face_decision])
                #print temp,scores
                scores=np.concatenate((scores,temp),axis=0)
    #print "bb\n",bboxes
    #print "s\n",scores
    bboxes=bboxes[1:,:]
    scores=scores[1:,:]
    return bboxes,scores
    raise Exception('Not Implemented Error')


'''
NON_MAX_SUPPRESSION Given a list of bounding boxes, returns a subset that
uses high confidence detections to suppresses other overlapping
detections. Detections can partially overlap, but the
center of one detection can not be within another detection.

Arguments:
    bboxes - ndarray of size (N,4) where N is the number of detections,
        and each row is [x_min, y_min, width, height]
    
    confidences - ndarray of size (N, 1) of the SVM confidence of each bounding
    	box.

    img_size - [height,width] dimensions of the image.

Returns:
    nms_bboxes -  ndarray of size (N, 4) where N is the number of non-overlapping
        detections, and each row is [x_min, y_min, width, height]. Each bounding box
        should not be overlapping significantly with any other bounding box.

In order to get the list of maximal bounding boxes, first sort bboxes by 
confidences. Then go through each of the bboxes in order, adding them to
the list if they do not significantly overlap with any already in the list. 
A significant overlap is if the center of one bbox is in the other bbox.
'''
def non_max_suppression(bboxes, confidences):
    # TODO: Implement this method!
    #print "confidences:\n",confidences
    idx=np.argsort(-confidences,axis=0)
    nms_bboxes=bboxes[idx[0,0]:idx[0,0]+1,:]
    window_h=nms_bboxes[0,3]
    window_w=nms_bboxes[0,2]
    #print "wid size:\n",window_h,window_w
    #print "idx\n",idx
    h,w=idx.shape
    #print "h:\n",h
    for i in xrange(1,h):
        temp=bboxes[idx[i,0]:idx[i,0]+1,:]
        hh,ww=nms_bboxes.shape
        temp1=np.matlib.repmat(temp,hh,1)
        #print "temp1:\n",temp1
        err=temp1-nms_bboxes
        err=np.abs(err[:,0:2])
        #err_max=np.min(err,axis=0)
        flag=np.logical_or(err[:,0]>window_w/2,err[:,1]>window_h/2)
        #print "flag:\n",flag
        f=flag[0]
        for ii in xrange(1,len(flag)):
            f=f&flag[ii]
        #flag1=[for ii in flag: True&ii]
        #idx_len=len(np.where(flag==True))
        #print "err min:\n",err_max
        if (f):
            nms_bboxes=np.concatenate((nms_bboxes,temp),axis=0)
            #print "nms bboxes:\n",nms_bboxes
    return nms_bboxes
    raise Exception('Not Implemented Error')


if __name__ == '__main__':
    block_size = 2
    cell_size = 6
    nbins = 9
    window_size = np.array([36, 36])

    # compute or load features for training
    if not (os.path.exists('data/features_pos.npy') and os.path.exists('data/features_neg.npy')):
        features_pos = get_positive_features('data/caltech_faces/Caltech_CropFaces', cell_size, window_size, block_size, nbins)
        num_negative_examples = 10000
        features_neg = get_random_negative_features('data/train_non_face_scenes', cell_size, window_size, block_size, nbins, num_negative_examples)
        np.save('data/features_pos.npy', features_pos)
        np.save('data/features_neg.npy', features_neg)
    else:
        features_pos = np.load('data/features_pos.npy')
        features_neg = np.load('data/features_neg.npy')

    X = np.vstack((features_pos, features_neg))
    Y = np.hstack((np.ones(len(features_pos)), np.zeros(len(features_neg))))

    # Train the SVM
    clf = LinearSVC(C=1, tol=1e-6, max_iter=10000, fit_intercept=True, loss='hinge')
    clf.fit(X, Y)
    score = clf.score(X, Y)

    # Part A: Sliding window detector
    im = imread('data/people.jpg', 'L').astype(np.uint8)
    bboxes, scores = run_detector(im, clf, window_size, cell_size, block_size, nbins)
    plot_img_with_bbox(im, bboxes, 'Without nonmaximal suppresion')
    plt.show()

    # Part B: Nonmaximal suppression
    bboxes = non_max_suppression(bboxes, scores)
    plot_img_with_bbox(im, bboxes, 'With nonmaximal suppresion')
    plt.show()
