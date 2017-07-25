"""CS231A Homework 1, Problem 3."""

import math
import numpy as np
import numpy.matlib
from utils import mat2euler


def compute_vanishing_point(points):
    """Computes vanishing point given four points on parallel line.

    Args:
        points: A list of all the points where each row is (x, y). Generally,
            it will contain four points: two for each parallel line.
            You can use any convention you'd like, but our solution uses the
            first two rows as points on the same line and the last
            two rows as points on the same line.
    Returns:
        vanishing_point: The pixel location of the vanishing point.
    """
    # TODO: Fill in this code.
    line1=points[:2,:]
    x1=line1[0,0]
    x2=line1[1,0]
    y1=line1[0,1]
    y2=line1[1,1]
    a1=y1-y2
    b1=x2-x1
    c1=-x2*(y1-y2)-y2*(x2-x1)
    dir1=np.array([a1,b1,c1])

    line1=points[2:,:]
    x1=line1[0,0]
    x2=line1[1,0]
    y1=line1[0,1]
    y2=line1[1,1]
    a1=y1-y2
    b1=x2-x1
    c1=-x2*(y1-y2)-y2*(x2-x1)
    dir2=np.array([a1,b1,c1])
    inte=np.cross(dir1,dir2)
    return inte[:2]/inte[2]
    pass

# It has some problme!!!!
def compute_K_from_vanishing_points(vanishing_points):
    """Compute intrinsic matrix given vanishing points.

    Args:
        vanishing_points: A list of vanishing points.
    Returns:
        K: The intrinsic camera matrix (3x3 matrix).
    """
    # TODO: Fill in this code.
    temp=np.ones((3,1))
    vanishing_points=np.concatenate((vanishing_points,temp),axis=1)
    x=vanishing_points[0,:]
    y=vanishing_points[1,:]
    a=x[0]*y[0]+x[1]*y[1]
    b=x[2]*y[0]+x[0]*y[2]
    c=x[2]*y[1]+x[1]*y[2]
    A1=np.array([[a,b,c]])
    b1=-x[2]*y[2]
    
    x=vanishing_points[0,:]
    y=vanishing_points[2,:]
    a=x[0]*y[0]+x[1]*y[1]
    b=x[2]*y[0]+x[0]*y[2]
    c=x[2]*y[1]+x[1]*y[2]
    A2=np.array([[a,b,c]])
    b2=-x[2]*y[2]

    x=vanishing_points[1,:]
    y=vanishing_points[2,:]
    a=x[0]*y[0]+x[1]*y[1]
    b=x[2]*y[0]+x[0]*y[2]
    c=x[2]*y[1]+x[1]*y[2]
    A3=np.array([[a,b,c]])
    b3=-x[2]*y[2]

    A=np.concatenate((A1,A2,A3),axis=0)
    b=np.array([[b1],[b2],[b3]])
    
    x=np.linalg.solve(A,b)
    #return x
    w1=x[0,0]
    w4=x[1,0]
    w5=x[2,0]
    W=np.array([[w1,0,w4],[0,w1,w5],[w4,w5,1]])
    #return W
    Winv=np.linalg.inv(W)
    #return Winv
    tx=Winv[0][2]
    ty=Winv[1][2]
    pixel_x=np.sqrt(Winv[0][0]-tx*tx)
    pixel_y=np.sqrt(Winv[1][1]-ty*ty)
    K=np.array([[pixel_x,0,tx],[0,pixel_y,ty],[0,0,1]])
    
    return K



    #l11=np.sqrt(Winv[0][0])
    #l21=Winv[1][0]/l11
    #l22=np.sqrt(Winv[1][1]-l21*l21)
    #l31=Winv[2][0]/l11
    #l32=(Winv[2][1]-l31*l21)/l22
    #l33=np.sqrt(Winv[2][2]-l31*l31-l32*l32)
    #K=np.array([[l11,l21,l31],[0,l22,l32],[0,0,l33]])
    #return K
    pass


def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    """Compute angle between planes of the given pairs of vanishing points.

    Args:
        vanishing_pair1: A list of a pair of vanishing points computed from
            lines within the same plane.
        vanishing_pair2: A list of another pair of vanishing points from a
            different plane than vanishing_pair1.
        K: The camera matrix used to take both images.
    Returns:
        angle: The angle in degrees between the planes which the vanishing
            point pair comes from2.
    """
    # TODO: Fill in this code.
    vanishing_pair1=np.concatenate((vanishing_pair1,np.ones((2,1))),axis=1)
    vanishing_pair2=np.concatenate((vanishing_pair2,np.ones((2,1))),axis=1)
    l1=np.cross(vanishing_pair1[0,:],vanishing_pair1[1,:])
    l2=np.cross(vanishing_pair2[0,:],vanishing_pair2[1,:])
    w=K.dot(K.T)                                   
    cosAngle=(l1.dot(w).dot(l2.T))/np.sqrt((l1.dot(w).dot(l1.T))*(l2.dot(w).dot(l2.T)))
    return math.acos(cosAngle)/math.pi*180                               
    pass


def compute_rotation_matrix_between_cameras(vanishing_pts1, vanishing_pts2, K):
    """Compute rotation matrix between two cameras given their vanishing points.

    Args:
        vanishing_pts1: A list of vanishing points in image 1.
        vanishing_pts2: A list of vanishing points in image 2.
        K: The camera matrix used to take both images.

    Returns:
        R: The rotation matrix between camera 1 and camera 2.
    """
    # TODO: Fill in this code.
    temp=np.ones((3,1))
    vanishing_pts1=np.concatenate((vanishing_pts1,temp),axis=1)
    vanishing_pts2=np.concatenate((vanishing_pts2,temp),axis=1)
    Kinv=np.linalg.inv(K)    
    vanishing_d1=Kinv.dot(vanishing_pts1.T)
    vanishing_d2=Kinv.dot(vanishing_pts2.T)
    mag_d1=np.sqrt(np.sum(np.square(vanishing_d1),axis=0))
    mag_d2=np.sqrt(np.sum(np.square(vanishing_d2),axis=0))
    mag_d1=np.matlib.repmat(mag_d1,3,1)
    mag_d2=np.matlib.repmat(mag_d2,3,1)
    vanishing_d1=np.divide(vanishing_d1,mag_d1)
    vanishing_d2=np.divide(vanishing_d2,mag_d2)
    R=np.linalg.inv(vanishing_d1).dot(vanishing_d2)
    return R
    pass


if __name__ == '__main__':
    # Part A: Compute vanishing points.
    v1 = compute_vanishing_point(np.array(
            [[674, 1826], [2456, 1060], [1094, 1340], [1774, 1086]]))
    v2 = compute_vanishing_point(np.array(
            [[674, 1826], [126, 1056], [2456, 1060], [1940, 866]]))
    v3 = compute_vanishing_point(np.array(
            [[1094, 1340], [1080, 598], [1774, 1086], [1840, 478]]))

    v1b = compute_vanishing_point(np.array(
            [[314, 1912], [2060, 1040], [750, 1378], [1438, 1094]]))
    v2b = compute_vanishing_point(np.array(
            [[314, 1912], [36, 1578], [2060, 1040], [1598, 882]]))
    v3b = compute_vanishing_point(np.array(
            [[750, 1378], [714, 614], [1438, 1094], [1474, 494]]))

    # Part B: Compute the camera matrix.
    vanishing_points = [v1, v2, v3]
    
    K_ours = compute_K_from_vanishing_points(vanishing_points)
    print "Intrinsic Matrix:\n", K_ours

    K_actual = np.array([[2448.0, 0, 1253.0], [0, 2438.0, 986.0], [0, 0, 1.0]])
    print
    print "Actual Matrix:\n", K_actual

    # Part D: Estimate the angle between the box and floor.
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array(
            [[1094, 1340], [1774, 1086], [1080, 598], [1840, 478]]))
    angle = compute_angle_between_planes(
            [floor_vanishing1, floor_vanishing2],
            [box_vanishing1, box_vanishing2], K_actual)
    print
    print "Angle between floor and box:", angle

    # Part E: Compute the rotation matrix between the two cameras.
    #print np.array([v1, v2, v3])
    rotation_matrix = compute_rotation_matrix_between_cameras(
            np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print
    print "Rotation between two cameras:\n", rotation_matrix
    z, y, x = mat2euler(rotation_matrix)
    x_angle = x * 180 / math.pi
    y_angle = y * 180 / math.pi
    z_angle = z * 180 / math.pi
    print
    print "Angle around z-axis (pointing out of camera): %f degrees" % z_angle
    print "Angle around y-axis (pointing vertically): %f degrees" % y_angle
    print "Angle around x-axis (pointing horizontally): %f degrees" % x_angle

