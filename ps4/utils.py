import os
from scipy.ndimage import imread
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_positive_features(train_path_pos, cell_size, window_size, block_size, nbins):
    '''
    'train_path_pos' is a string. This directory contains 36x36 images of
      faces
    'feature_params' is a struct, with fields
      feature_params.template_size (probably 36), the number of pixels
         spanned by each train / test template and
      feature_params.hog_cell_size (default 6), the number of pixels in each
         HoG cell. template size should be evenly divisible by hog_cell_size.
         Smaller HoG cell sizes tend to work better, but they make things
         slower because the feature dimensionality increases and more
         importantly the step size of the classifier decreases at test time.

    'features_pos' is N by D matrix where N is the number of faces and D
    is the template dimensionality, which would be
      (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
    if you're using the default vl_hog parameters
    '''

    image_files = [os.path.join(train_path_pos, f) for f in os.listdir(train_path_pos) if f.endswith('.jpg')]
    num_images = len(image_files)
    total_block_size = block_size * cell_size
    template_size = int((np.floor((window_size[0] - 2) / (total_block_size / 2)) - 1) * (np.floor((window_size[1] - 2) / (total_block_size / 2)) - 1))
    D = template_size * block_size * block_size * nbins
    features_pos = np.zeros((num_images, D))
    for i in range(num_images):
        img = imread(image_files[i])
        features_pos[i] = compute_hog_features(img, cell_size, block_size, nbins).reshape(-1)
    return features_pos

def get_random_negative_features(non_face_scn_path, cell_size, window_size, block_size, nbins, num_samples):
    '''
    'non_face_scn_path' is a string. This directory contains many images
      which have no faces in them.
    'feature_params' is a struct, with fields
      feature_params.template_size (probably 36), the number of pixels
         spanned by each train / test template and
      feature_params.hog_cell_size (default 6), the number of pixels in each
         HoG cell. template size should be evenly divisible by hog_cell_size.
         Smaller HoG cell sizes tend to work better, but they make things
         slower because the feature dimensionality increases and more
         importantly the step size of the classifier decreases at test time.
    'num_samples' is the number of random negatives to be mined, it's not
      important for the function to find exactly 'num_samples' non-face
      features, e.g. you might try to sample some number from each image, but
      some images might be too small to find enough.

    'features_neg' is N by D matrix where N is the number of non-faces and D
    is the template dimensionality, which would be
      (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
    if you're using the default vl_hog parameters
    '''

    image_files = [os.path.join(non_face_scn_path, f) for f in os.listdir(non_face_scn_path) if f.endswith('.jpg')]
    num_images = len(image_files)
    num_sample_per_img = int(np.ceil(num_samples * 1. / num_images))
    total_block_size = block_size * cell_size
    template_size = [int(np.floor((window_size[0] - 2) / (total_block_size / 2)) - 1), int(np.floor((window_size[1] - 2) / (total_block_size / 2)) - 1)]
    D = template_size[0] * template_size[1] * block_size * block_size * nbins
    features_neg = np.zeros((num_images * num_sample_per_img, D))
    for i in range(num_images):
        img = imread(image_files[i], 'L').astype(np.uint8)
        height, width = img.shape
        for j in range(num_sample_per_img):
            top_left_x = int(np.ceil(np.random.rand() * (width - window_size[1])))
            top_left_y = int(np.ceil(np.random.rand() * (height - window_size[0])))
            index = i * num_sample_per_img + j
            cropped = img[top_left_y : top_left_y + window_size[0], top_left_x : top_left_x + window_size[1]]
            features_neg[index,...] = compute_hog_features(cropped, cell_size, block_size, nbins).reshape(-1)
    return features_neg


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
    H, W = im.shape
    xgrad = np.zeros((H-2, W-2))
    ygrad = np.zeros((H-2, W-2))

    xgrad = im[1:-1, :-2] - im[1:-1, 2:]
    ygrad = im[:-2, 1:-1] - im[2:, 1:-1]

    angles = np.arctan2(ygrad, xgrad)
    angles[angles < 0] += math.pi
    angles = np.degrees(angles)
    magnitudes = np.sqrt(xgrad ** 2 + ygrad ** 2)
    return angles, magnitudes
    

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
    histogram = np.zeros(nbins)

    bin_size = float(180 / nbins)

    # iterate over the pixels
    for h in xrange(angles.shape[0]):
        for w in xrange(angles.shape[1]):
            ang = angles[h,w]
            mag = magnitudes[h,w]

            if ang >= 180:
                ang = ang - 180

            # interpolate the votes
            lower_idx = int(ang / bin_size) - 1
            upper_idx = lower_idx + 1

            lower_ang = lower_idx * bin_size + 90/nbins
            upper_ang = upper_idx * bin_size + 90/nbins

            # Account for edge case
            if upper_idx >= nbins:
                upper_idx = 0
            if lower_idx < 0:
                lower_idx = nbins - 1

            lower_diff= abs(ang - lower_ang)
            upper_diff = abs(ang - upper_ang)
            lower_percent = upper_diff/ bin_size
            upper_percent = lower_diff/ bin_size
            histogram[lower_idx] += lower_percent * mag
            histogram[upper_idx] += upper_percent * mag

    return histogram


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
    height = im.shape[0] - 2
    width = im.shape[1] - 2 

    angles, magnitudes = compute_gradient(im)

    total_cells_in_block = cells_in_block * pixels_in_cell
    stride = total_cells_in_block / 2
    features = np.zeros((int(math.floor(height / stride)) - 1, 
        int(math.floor(width / stride)) - 1, 
        nbins * cells_in_block * cells_in_block))

    # iterate over the blocks, 50% overlap
    for w in xrange(0, width - total_cells_in_block, stride):
        for h in xrange(0, height - total_cells_in_block, stride):
            block_features = np.zeros((cells_in_block, cells_in_block,  nbins))
            block_magnitude = magnitudes[h:h+total_cells_in_block, 
                w:w+total_cells_in_block]
            block_angle = angles[h:h+total_cells_in_block,
                    w:w+total_cells_in_block]
            
            #  iterate over the cells
            for i in xrange(cells_in_block):
                for j in xrange(cells_in_block):
                    cell_magnitudes = block_magnitude[i * pixels_in_cell:(i+1)
                            * pixels_in_cell,
                            j*pixels_in_cell:(j+1)*pixels_in_cell]
                    cell_angles = block_angle[i * pixels_in_cell:(i+1)
                            * pixels_in_cell,
                            j*pixels_in_cell:(j+1)*pixels_in_cell]
                    block_features[i,j,:] = generate_histogram(cell_angles,
                            cell_magnitudes, nbins)
                           
            block_features = block_features.flatten()
            block_features = block_features \
                / np.sqrt(np.linalg.norm(block_features) ** 2 + .01)
            features[int(math.ceil(h/(stride))),
                int(math.ceil(w/(stride))),:] = block_features

    return features


# Displays the HoG features next to the original image
def plot_img_with_bbox(im, bbox, title_text = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(bbox.shape[0]):
        ax.add_patch(
            patches.Rectangle(
                (bbox[i,0], bbox[i,1]),
                bbox[i,2],
                bbox[i,3],
                fill=False,
                edgecolor='red'
            )
        )
    plt.imshow(im, 'gray')
    if title_text is not None:
        plt.title(title_text)
