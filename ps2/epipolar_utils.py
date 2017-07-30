import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
GET_DATA_FROM_TXT_FILE
Arguments:
    filename - a path (str) to the data location
Returns:
    points - a matrix of points where each row is either:
        a) the homogenous coordinates (x,y,1) if the data is 2D
        b) the coordinates (x,y,z) if the data is 3D
    use_subset - use a predefined subset (this is hard coded for now)
'''
def get_data_from_txt_file(filename, use_subset = False):
    with open(filename) as f:
            lines = f.read().splitlines()
    number_pts = int(lines[0])

    points = np.ones((number_pts, 3))
    for i in xrange(number_pts):
        split_arr = lines[i+1].split()
        if len(split_arr) == 2:
            y, x = split_arr
        else:
            x, y, z = split_arr
            points[i,2] = z
        points[i,0] = x 
        points[i,1] = y
    return points

'''
COMPUTE_RECTIFIED_IMAGE
Arguments:
    im - an image
    H - a homography matrix that rectifies the image
Returns:
    new_image - a new image matrix after applying the homography
    offset - the offest in the image.
'''
def compute_rectified_image(im, H):
    new_x = np.zeros(im.shape[:2])
    new_y = np.zeros(im.shape[:2])
    for y in xrange(im.shape[0]): # height
        for x in xrange(im.shape[1]): # width
            new_location = H.dot([x, y, 1])
            new_location /= new_location[2]
            new_x[y,x] = new_location[0]
            new_y[y,x] = new_location[1]
    offsets = (new_x.min(), new_y.min())
    new_x -= offsets[0]
    new_y -= offsets[1]
    new_dims = (int(np.ceil(new_y.max()))+1,int(np.ceil(new_x.max()))+1)

    H_inv = np.linalg.inv(H)
    new_image = np.zeros(new_dims)

    for y in xrange(new_dims[0]):
        for x in xrange(new_dims[1]):
            old_location = H_inv.dot([x+offsets[0], y+offsets[1], 1])
            old_location /= old_location[2]
            old_x = int(old_location[0])
            old_y = int(old_location[1])
            if old_x >= 0 and old_x < im.shape[1] and old_y >= 0 and old_y < im.shape[0]:
                new_image[y,x] = im[old_y, old_x]

    return new_image, offsets

'''
SCATTER_3D_AXIS_EQUAL
Arguments:
    X - the coordinates on the x axis (N long vector)
    Y - the coordinates on the y axis (N long vector)
    Z - the coordinates on the z axis (N long vector)
    ax - the pyplot axis
Returns:
    Nothing; instead plots the points of (X, Y, Z) such that the axes are equal
'''
def scatter_3D_axis_equal(X, Y, Z, ax):
    ax.scatter(X, Y, Z)
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
