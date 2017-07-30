import numpy as np
import os
import scipy.misc
from scipy.optimize import least_squares
import math
from copy import deepcopy
from triangulation import *

# A class that stores information about multiple views of the scene
class Frame:
    def __init__(self, matches, focal_length, F, im_width, im_height):
        self.focal_length = focal_length
        self.im_height = im_height
        self.im_width = im_width
        self.matches = matches

        self.N = matches.shape[0]
        self.match_idx = np.array([np.arange(self.N), 
            np.arange(self.N, 2 * self.N)])
        self.match_points = np.vstack((matches[:,:2], matches[:,2:]))

        self.K = np.eye(3)
        self.K[0,0] = self.K[1,1] = focal_length
        self.E = self.K.T.dot(F).dot(self.K)
        self.T = estimate_RT_from_E(self.E, matches.reshape((-1,2,2)), self.K)

        self.motion = np.zeros((2,3,4))
        self.motion[0,:,:-1] = np.eye(3)
        self.motion[1,:,:] = self.T
        self.structure = triangulate(self)

'''
NEG_ONES
Arguments:
    size - the dimensions of a tensor given as a tuple or array
    dtype - the numpy type of the tensor
Returns:
    neg_ones - similar to np.zeros or np.ones, but with -1
'''
def neg_ones(size, dtype = np.int16):
    return -1 * np.ones(size, dtype = dtype)

'''
TRIANGULATE
Arguments:
    frame - information about multiple views of the scene
Returns:
    structure - the 3D points of the scene
'''
def triangulate(frame):
    num_cameras, num_points = frame.match_idx.shape
    structure = np.zeros((num_points,3))
    all_camera_matrices = np.zeros((num_cameras, 3, 4))
    for i in xrange(num_cameras):
        all_camera_matrices[i,:,:] = frame.K.dot(frame.motion[i,:,:])
    for i in xrange(num_points):
        valid_cameras = np.where(frame.match_idx[:,i] >= 0)[0]
        camera_matrices = all_camera_matrices[valid_cameras,:,:]
        x = np.zeros((len(valid_cameras), 2))
        for ctr, c in enumerate(valid_cameras):
            x[ctr, :] = frame.match_points[frame.match_idx[c, i],:]
        structure[i,:] = nonlinear_estimate_3d_point(x, camera_matrices)

    return structure 

'''
ROTATION_MATRIX_TO_ANGLE_AXIS
Arguments:
    R - a rotation matrix
Returns:
    angle_axis - the angle axis representation of the rotation
'''
def rotation_matrix_to_angle_axis(R):
    angle_axis = np.array([0.0]*3)
    angle_axis[0] = R[2, 1] - R[1, 2]
    angle_axis[1] = R[0, 2] - R[2, 0]
    angle_axis[2] = R[1, 0] - R[0, 1]
    
    cos_theta = min(max((R[0,0]+R[1,1]+R[2,2] - 1.0) / 2.0, -1.0), 1.0)
    sin_theta = min(np.sqrt((angle_axis**2).sum())/2, 1.0);

    theta = math.atan2(sin_theta, cos_theta)

    k_threshold = 1e-12
    if ((sin_theta > k_threshold) or (sin_theta < -k_threshold)):
        r = theta / (2.0 * sin_theta)
        angle_axis = angle_axis * r
        return angle_axis

    if cos_theta > 0:
        angle_axis = angle_axis / 2
        return angle_axis

    inv_one_minus_cos_theta = 1.0 / (1.0 - cos_theta)

    for i in xrange(3):
        angle_axis[i] = theta * math.sqrt((R[i,i] - cos_theta) 
            * inv_one_minus_cos_theta)
        if((sin_theta < 0 and angle_axis[i] > 0) or
            (sin_theta > 0 and angle_axis[i] < 0)):
            angle_axis[i] *= -1

    return angle_axis

'''
ROTATION_MATRIX_TO_ANGLE_AXIS
Arguments:
    angle_axis - the angle axis representation of the rotation
Returns:
    R - the corresponding rotation matrix
'''
def angle_axis_to_rotation_matrix(angle_axis):
    theta2 = np.dot(angle_axis, angle_axis)
    R = np.zeros((3,3))
    if theta2 > 0:
        theta = np.sqrt(theta2)
        wx, wy, wz = tuple(angle_axis / theta)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        R[0, 0] = cos_theta + wx * wx * (1 - cos_theta)
        R[1, 0] = wz * sin_theta + wx * wy * (1 - cos_theta)
        R[2, 0] = -wy * sin_theta + wx * wz * (1 - cos_theta)
        R[0, 1] =  wx * wy * (1 - cos_theta) - wz * sin_theta;
        R[1, 1] = cos_theta   + wy * wy * (1 - cos_theta);
        R[2, 1] =  wx * sin_theta   + wy * wz * (1 - cos_theta);
        R[0, 2] =  wy * sin_theta   + wx * wz * (1 - cos_theta);
        R[1, 2] = -wx * sin_theta   + wy * wz * (1 - cos_theta);
        R[2, 2] = cos_theta   + wz * wz * (1 - cos_theta);

    else:
        # At zero, we switch to using the first order Taylor expansion.
        R[0, 0] =  1;
        R[1, 0] = -angle_axis[2];
        R[2, 0] =  angle_axis[1];
        R[0, 1] =  angle_axis[2];
        R[1, 1] =  1;
        R[2, 1] = -angle_axis[0];
        R[0, 2] = -angle_axis[1];
        R[1, 2] =  angle_axis[0];
        R[2, 2] = 1;
    return R

'''
CROSS_PRODUCT_MAT
Arguments:
    a - a 3x1 vector
Returns:
    m - the corresponding cross-product matrix [a]_x
'''
def cross_product_mat(a):
    m = np.zeros((3,3))
    m[1,0] = a[2]
    m[0,1] = -a[2]
    m[2,0] = -a[1]
    m[0,2] = a[1]
    m[2,1] = a[0]
    m[1,2] = -a[0]
    return m

'''
ANGLE_AXIS_ROTATE
Arguments:
    angle_axis - the angle axis representation of the rotation
    pt - a matrix containing points
Returns:
    new_pts - a matrix containing the points after the rotation is applied
'''
def angle_axis_rotate(angle_axis, pt):
    aa = angle_axis[:3].reshape((1,3))
    theta2 = aa.dot(aa.T)[0,0]
    if theta2 > 0:
        theta = np.sqrt(theta2)
        w = aa / theta
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        w_cross_pt = cross_product_mat(w[0]).dot(pt)

        w_dot_pt = w.dot(pt)

        result = pt * cos_theta + w_cross_pt * sin_theta + (w.T * (1 - cos_theta)).dot(w_dot_pt)
    else:
        w_cross_pt = cross_product_mat(aa[0,:]).dot(pt)
        result = pt + w_cross_pt
    return result
        

'''
REPROJECTION_ERROR_MOT_STR
Arguments:
    match_idx - a MxN matrix corresponding to what indices of the match_points
        are visible in each of the M cameras
    match_points - the pixel locations in the image
    f - the focal length
    px - the principal component in the x direction
    py - the principal component in the y direction
    motion - the camera's motion tensor
    structure - the current 3D point estimate of the scene
Returns:
    errors - the reprojection error vector
'''
def reprojection_error_mot_str(match_idx, match_points, f, px, py, motion, structure):
    N = match_idx.shape[0]

    errors = None
    for i in xrange(N):
        valid_pts = match_idx[i,:] >= 0
        valid_idx = match_idx[i, valid_pts]

        RP = angle_axis_rotate(motion[i, :, 0], structure[valid_pts,:].T)

        TRX = RP[0, :] + motion[i, 0, 1]
        TRY = RP[1, :] + motion[i, 1, 1]
        TRZ = RP[2, :] + motion[i, 2, 1]

        TRXoZ = TRX / TRZ
        TRYoZ = TRY / TRZ
        
        x = f * TRXoZ + px
        y = f * TRYoZ + py

        ox = match_points[valid_idx, 0]
        oy = match_points[valid_idx, 1]

        if errors is None:
            errors = np.vstack((x-ox, y-oy))
        else:
            errors = np.hstack((errors, np.vstack((x-ox, y-oy))))

    return errors.flatten()
'''
REPROJECTION_ERROR_MOT_STR_OPT
Arguments:
    mot_str - the motion and structure matrices flattened into one vector
        this allows us to use the builtin nonlinear optimization methods
    match_idx - a MxN matrix corresponding to what indices of the match_points
        are visible in each of the M cameras
    match_points - the pixel locations in the image
    f - the focal length
    px - the principal component in the x direction
    py - the principal component in the y direction
Returns:
    errors - the reprojection error vector
'''
def reprojection_error_mot_str_opt(mot_str, match_idx, match_points, f, px, py):
    num_cameras = match_idx.shape[0]
    cut = 3 * 2 * num_cameras
    structure = mot_str[cut:].reshape((-1,3))
    motion = mot_str[:cut].reshape((-1,3,2))

    error = reprojection_error_mot_str(match_idx, match_points, f, px, py, motion, structure)
    return error

'''
BUNDLE_ADJUSTMENT
Arguments:
    frame - the information about multiple views of the scene
Returns:
    Nothing; updates the motion and structure information
'''
def bundle_adjustment(frame):
    num_cameras = frame.motion.shape[0]
    motion_angle_axis = np.zeros((num_cameras, 3, 2))
    for i in xrange(num_cameras):
        motion_angle_axis[i, :, 0] = rotation_matrix_to_angle_axis(
                frame.motion[i,:, :-1])
        motion_angle_axis[i, :, 1] = frame.motion[i, :, -1]

    px = 0
    py = 0
    
    errors = reprojection_error_mot_str(frame.match_idx, frame.match_points, frame.focal_length, px, py, motion_angle_axis, frame.structure)

    vec = least_squares(reprojection_error_mot_str_opt, np.hstack((motion_angle_axis.flatten(), frame.structure.flatten())),
        args=(frame.match_idx, frame.match_points, frame.focal_length, px, py), method='lm')

    cut = 3 * 2 * num_cameras

    opt_val = vec['x']
    frame.structure = opt_val[cut:].reshape((-1,3))
    motion_angle_axis = opt_val[:cut].reshape((-1, 3, 2))

    for i in xrange(num_cameras):
        frame.motion[i,:,:] = np.hstack((angle_axis_to_rotation_matrix(motion_angle_axis[i,:,0]), motion_angle_axis[i,:,1].reshape((3,1))))

'''
MULTIPLY_TRANSFORMATIONS
Arguments:
    A - a matrix
    B - another matrix
Returns:
    M - the matrix that occurs when you rotate B by A's rotation component
        and then translate by A's translation component
'''
def multiply_transformations(A, B):
    return np.hstack((A[:,:3].dot(B[:,:3]), (A[:,:3].dot(B[:,-1]) + A[:,-1]).reshape((3,-1))))

'''
INVERSE
Arguments:
    x - a 3x4 matrix that assumes only rotation and translation
Returns:
    inv_x - the 3x4 matrix inverse of x
'''
def inverse(x):
    return np.hstack((x[:3, :3].T, -x[:3, :3].T.dot(x[:3, -1]).reshape((3,-1))))

'''
TRANSFORM_POINTS
Arguments:
    points_3d - 3d points listed as a 3xN matrix
    Rt - the 3x4 transformation matrix
    is_inverse - whether to use the inverted Rt or non-inverted
Returns:
    new_points - the points after applying the Rt
'''
def transform_points(points_3d, Rt, is_inverse = False):
    if is_inverse:
        return Rt[:,:3].T.dot((points_3d - Rt[:,-1]).T).T
    return Rt[:,:3].dot(points_3d.T).T + Rt[:,-1]

'''
ROW_INTERSECTION
Arguments:
    A - a matrix
    B - another matrix
Returns:
    intersect - a matrix whose rows are both rows in A and B
    idA - indices of these rows in A
    idB - indices of these rows in B
'''
def row_intersection(A, B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
                   'formats':ncols * [A.dtype]}
    intersect = np.intersect1d(A.view(dtype), B.view(dtype))
    intersect = intersect.view(A.dtype).reshape((-1,ncols))
    idA = np.array([np.where(np.all(A==x, axis=1))[0][0] for x in intersect])
    idB = np.array([np.where(np.all(B==x, axis=1))[0][0] for x in intersect])
    return intersect, idA, idB

'''
ROW_SET_DIFF
Arguments:
    A - a matrix
    B - another matrix
Returns:
    set_diff - a matrix whose rows are only in one of A or B
    idA - indices of these rows in A
    idB - indices of these rows in B
'''
def row_set_diff(A, B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
                   'formats':ncols * [A.dtype]}
    set_diff = np.setdiff1d(A.view(dtype), B.view(dtype))
    set_diff = set_diff.view(A.dtype).reshape((-1,ncols))
    idA = []
    idB = []
    for x in set_diff:
        idx_in_A = np.where(np.all(A==x, axis=1))[0]
        idx_in_B = np.where(np.all(B==x, axis=1))[0]
        if len(idx_in_A) != 0:
            idA.append(idx_in_A[0])
        if len(idx_in_B) != 0:
            idB.append(idx_in_B[0])

    return set_diff, np.array(idA), np.array(idB)

'''
REMOVE_OUTLIERS
Arguments:
    frame - the information about multiple views of the scene
    threshold - a threshold for the sum square of the reprojection error
Returns:
    Nothing; updated frame information (structure)
'''
def remove_outliers(frame, threshold = 10.0):
    threshold *= threshold
    threshold_in_degree = 2.0
    threshold_in_cos = math.cos(float(threshold_in_degree) / 180 * math.pi)

    for i in xrange(frame.match_idx.shape[0]):
        X = frame.K.dot(transform_points(frame.structure, frame.motion[i,:,:]).T)
        xy = X[:2, :] / X[2, :]
        selector = np.where(frame.match_idx[i,:] >= 0)[0]
        diff = xy[:, selector].T - frame.match_points[frame.match_idx[i, selector],:]
        outliers = np.sum(diff**2, axis=1) > threshold
        
        pts2keep = np.array([True] * frame.structure.shape[0])
        pts2keep[selector[outliers]] = False

        frame.structure = frame.structure[pts2keep, :]
        frame.match_idx = frame.match_idx[:, pts2keep]

    # check viewing angle
    num_frames = frame.motion.shape[0]
    positions = np.zeros((3, num_frames))
    for i in xrange(num_frames):
        Rt = frame.motion[i, : , :]
        positions[:, i] = -Rt[:3, :3].T.dot(Rt[:,-1])
    
    view_dirs = np.zeros((3, frame.structure.shape[0], num_frames))
    for i in xrange(frame.match_idx.shape[0]-1):
        selector = np.where(frame.match_idx[i,:] >= 0)[0]
        camera_view_dirs = frame.structure[selector,:] - positions[:, i]
        dir_length = np.sqrt(np.sum(camera_view_dirs ** 2))
        camera_view_dirs = camera_view_dirs / dir_length
        view_dirs[:, selector, i] = camera_view_dirs.T

    for c1 in xrange(num_frames):
        for c2 in xrange(c1,num_frames):
            if c1 == c2: continue
            selector1 = np.where(frame.match_idx[c1,:] >= 0)[0]
            selector2 = np.where(frame.match_idx[c2,:] >= 0)[0]
            selector = np.array([x for x in selector1 if x in selector2])
            if len(selector) == 0:
                continue
            view_dirs_1 = view_dirs[:, selector, c1]
            view_dirs_2 = view_dirs[:, selector, c2]
            cos_angles = np.sum(view_dirs_1 * view_dirs_2, axis=0)
            outliers = cos_angles > threshold_in_cos

            pts2keep = np.array([True] * frame.structure.shape[0])
            pts2keep[selector[outliers]] = False
            frame.structure = frame.structure[pts2keep, :]
            frame.match_idx = frame.match_idx[:,pts2keep]

'''
MERGE_TWO_FRAMES
Arguments:
    frameA - a frame 
    frameB - another frame, we assume the first index of frameB corresponds to
        the last camera index of frameA
    length - the number of total cameras
Returns:
    merged_frame - the merged information of both frames
'''
def merge_two_frames(frameA, frameB, length):
    merged_frame = deepcopy(frameA)

    frameB_to_A = multiply_transformations(inverse(frameA.motion[-1,:,:]), frameB.motion[0,:,:])
    frameB.structure = transform_points(frameB.structure, frameB_to_A)
    for i in xrange(2):
        frameB.motion[i,:,:] = multiply_transformations(frameB.motion[i,:,:], inverse(frameB_to_A))

    # since camera is in merged reference frame, add it to motion matrix
    merged_frame.motion = np.vstack((merged_frame.motion, frameB.motion[-1,:,:].reshape((-1,3,4))))

    # we need to reconcile the matched points to generate the structure.
    # we need to merge the matched_points from each additional frame, but must 
    # associate points that correspond to already seen points in the same
    # column
    trA = np.where(frameA.match_idx[0,:] >= 0)[0]
    xyA = frameA.match_points[frameA.match_idx[-1, trA], :]

    trB = np.where(frameB.match_idx[0,:] >= 0)[0]
    xyB = frameB.match_points[frameB.match_idx[0, trB], :]

    xy_common, iA, iB = row_intersection(xyA, xyB)
    xy_common = xy_common.T

    merged_frame.match_idx = np.vstack((merged_frame.match_idx, neg_ones((1, merged_frame.match_idx.shape[1]))))
    for i in xrange(xy_common.shape[1]):
        idA = trA[iA[i]]
        idB = trB[iB[i]]

        B_match_idx = frameB.match_idx[1, idB]

        merged_frame.match_points = np.vstack((merged_frame.match_points, frameB.match_points[B_match_idx, :]))
        merged_frame.match_idx[length, idA] = merged_frame.match_points.shape[0]-1

    # One of the cameras in frame B is the same as frame A
    # We will add all new points from this camera into the match fields
    xy_new, iB, iA = row_set_diff(xyB, xyA)
    xy_new = xy_new.T

    for i in xrange(xy_new.shape[1]):
        idB = trB[iB[i]]

        merged_frame.match_points = np.vstack((merged_frame.match_points,frameB.match_points[frameB.match_idx[0,idB],:]))
        merged_frame.match_idx = np.hstack((merged_frame.match_idx, neg_ones((merged_frame.match_idx.shape[0],1))))
        merged_frame.match_idx[length-1,-1] = merged_frame.match_points.shape[0]-1
        merged_frame.structure = np.vstack((merged_frame.structure, frameB.structure[idB,:]))

        B_match_idx = frameB.match_idx[1, idB]

        merged_frame.match_points = np.vstack((merged_frame.match_points, frameB.match_points[B_match_idx,:]))
        merged_frame.match_idx[length,-1] = merged_frame.match_points.shape[0]-1

    # The other camera in frame B is new
    # We can simply add all the points from here
    # TODO: This part is not implemented currently, but a full SFM pipeline should have it 

    return merged_frame

'''
MERGE_ALL_FRAMES
Arguments:
    frames - a list of frames
Returns:
    merged_frame - a frame which contains the information of all the frames
'''
def merge_all_frames(frames):
    merged_frame = deepcopy(frames[0])
    for i in xrange(1,len(frames)):
        merged_frame = merge_two_frames(merged_frame, frames[i], i+1)
        merged_frame.structure = triangulate(merged_frame)
        bundle_adjustment(merged_frame)
        remove_outliers(merged_frame, 10)
        bundle_adjustment(merged_frame)

    return merged_frame
