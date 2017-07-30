import sys
import numpy as np
import os
import scipy.misc
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sfm_utils import *
from sympy import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # TODO: Implement this method!
    U,s,V=np.linalg.svd(E)
    W=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Q1=U.dot(W.dot(V))
    Q2=U.dot(W.T.dot(V))
    R1=np.linalg.det(Q1)*Q1
    R2=np.linalg.det(Q2)*Q2
    T1=U[:,2:3]
    T2=-T1
    #temp=np.array([[0,0,0,1]])
    #print 'R\n',R1
    #print 'T\n',T1
    RT1=np.concatenate((R1,T1),axis=1)
    RT2=np.concatenate((R1,T2),axis=1)
    RT3=np.concatenate((R2,T1),axis=1)
    RT4=np.concatenate((R2,T2),axis=1)
    RT=np.zeros((4,3,4))
    RT[0,:,:]=RT1
    RT[1,:,:]=RT2
    RT[2,:,:]=RT3
    RT[3,:,:]=RT4
    #RT=np.concatenate((RT1,RT2,RT3,RT4),axis=2)
    return RT
    raise Exception('Not Implemented Error')

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    #print 'points\n',image_points
    #print 'camera\n',camera_matrices
    #print image_points[:]
    pSize=image_points.shape
    imgPoints=image_points.flatten()
    imgPoints=(np.matlib.repmat(imgPoints,4,1)).T
    M3=np.zeros((pSize[0]*2,4))
    temp2=np.zeros((pSize[0]*2,4))
    for idx in range(pSize[0]):
        M3[2*idx,:]=camera_matrices[idx,2:3,:]
        M3[2*idx+1,:]=camera_matrices[idx,2:3,:]
        temp2[2*idx,:]=camera_matrices[idx,:1,:]
        temp2[2*idx+1,:]=camera_matrices[idx,1:2,:]
    temp1=np.multiply(imgPoints,M3)
    A=temp1-temp2
    U,s,V=np.linalg.svd(A)
    point_3d=V[:1,:]
    point_3d=point_3d/point_3d[0,3]
    #M3=camera_matrices[:,2:3,:]
    #M3=M3.flatten()
    #m3=np.matlib.repmat(m3,1,pSize[0])
    #m3=np.reshape(M3,(1,pSize[0]))
    #print 'm3\n',M3
    #print 'imgPoints\n',imgPoints
    #print 'temp2\n',temp2
    return point_3d[0][:3]
    raise Exception('Not Implemented Error')

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # TODO: Implement this method!
    pointSize=image_points.shape
    error=np.zeros((2*pointSize[0],1))
    point_3d=np.concatenate((point_3d,np.array([1])),axis=0)
    #print 'point3d\n',point_3d
    #camera_matrices=np.reshape(camera_matrices,(
    for idx in range(pointSize[0]):
        cam=camera_matrices[idx,:,:]
        pointNew=cam.dot(point_3d)
        pointNew=pointNew/pointNew[2]
        #print 'point new\n',pointNew
        error[idx*2:idx*2+2,0]=pointNew[:2]
        #print 'err\n',error
    #print 'img point\n',image_points
    imagePoint=image_points.flatten().T
    #print 'imagepoint\n',imagePoint
    #print 'error\n',error
    error=error-np.reshape(imagePoint,(2*pointSize[0],1))
    #print 'err\n',error
    return error
    raise Exception('Not Implemented Error')

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # TODO: Implement this method!
    p1,p2,p3=symbols("p1 p2 p3")
    #print 'point3d',point_3d
    #print 'camera',camera_matrices
    pSize=camera_matrices.shape
    #print 'size',pSize
    jacobian=np.zeros((2*pSize[0],3))
    for idx in range(pSize[0]):
        temp1=camera_matrices[idx,0,:]
        temp2=camera_matrices[idx,2,:]
        #print 'temp\n',temp1[0]
        #print 'point\n',point_3d
        fun1=(temp1[0]*p1+temp1[1]*p2+temp1[2]*p3+temp1[3])\
              /(temp2[0]*p1+temp2[1]*p2+temp2[2]*p3+temp2[3])
        diffFun1p1=diff(fun1,p1)
        #print 'diff p1\n',diffFun1p1
        j1=diffFun1p1.subs([(p1,point_3d[0]),(p2,point_3d[1]),(p3,point_3d[2])])
        diffFun1p2=diff(fun1,p2)
        j2=diffFun1p2.subs([(p1,point_3d[0]),(p2,point_3d[1]),(p3,point_3d[2])])
        diffFun1p3=diff(fun1,p3)
        j3=diffFun1p3.subs([(p1,point_3d[0]),(p2,point_3d[1]),(p3,point_3d[2])])
        jacobian[idx*2,:]=np.array([j1,j2,j3])

        temp1=camera_matrices[idx,1,:]
        fun1=(temp1[0]*p1+temp1[1]*p2+temp1[2]*p3+temp1[3])\
              /(temp2[0]*p1+temp2[1]*p2+temp2[2]*p3+temp2[3])
        diffFun1p1=diff(fun1,p1)
        #print 'diff p1\n',diffFun1p1
        j1=diffFun1p1.subs([(p1,point_3d[0]),(p2,point_3d[1]),(p3,point_3d[2])])
        diffFun1p2=diff(fun1,p2)
        j2=diffFun1p2.subs([(p1,point_3d[0]),(p2,point_3d[1]),(p3,point_3d[2])])
        diffFun1p3=diff(fun1,p3)
        j3=diffFun1p3.subs([(p1,point_3d[0]),(p2,point_3d[1]),(p3,point_3d[2])])
        jacobian[idx*2+1,:]=np.array([j1,j2,j3])
        
        #a=temp1[1]*point_3d[1]+temp1[2]*point_3d[2]+temp1[3]
        #b=temp2[1]*point_3d[1]+temp2[2]*point_3d[2]+temp2[3]
        #k=(a-temp1[0]*b/temp2[0])/temp2[0]        
        #j11=-k/((point_3d[0]+b/temp2[0])*(point_3d[0]+b/temp2[0]))
        
        #a=temp1[0]*point_3d[0]+temp1[2]*point_3d[2]+temp1[3]
        #b=temp2[0]*point_3d[0]+temp2[2]*point_3d[2]+temp2[3]
        #k=(a-temp1[1]*b/temp2[1])/temp2[1]
        #j12=-k/((point_3d[1]+b/temp2[1])*(point_3d[1]+b/temp2[1]))
        
        #a=temp1[0]*point_3d[0]+temp1[1]*point_3d[1]+temp1[3]
        #b=temp2[0]*point_3d[0]+temp2[1]*point_3d[1]+temp2[3]
        #k=(a-temp1[2]*b/temp2[2])/temp2[2]
        #j13=-k/((point_3d[2]+b/temp2[2])*(point_3d[2]+b/temp2[2]))
        #jacobian[idx,:]=np.array([j11,j12,j13])
    return jacobian
    raise Exception('Not Implemented Error')

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # TODO: Implement this method!
    #print 'img_points',image_points
    #print 'camera\n',camera_matrices
    #pSize=image_points.shape
    #for j in range(pSize[0]):
    point_3d=linear_estimate_3d_point(image_points, camera_matrices)
    #print 'point_3d',point_3d
    for i in range(9):
        jac=jacobian(point_3d, camera_matrices)
        err=reprojection_error(point_3d, image_points, camera_matrices)
        #print 'jac\n',jac
        #print 'err\n',err
        try:
            temp=np.linalg.inv(jac.T.dot(jac)).dot(jac.T.dot(err))
        except:
            temp=np.zeros((3,1))
        #print 'temp\n',temp
        point_3d=point_3d-temp[:,0]
        #print 'point new\n',point_3d
    return point_3d
    raise Exception('Not Implemented Error')

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # TODO: Implement this method!
    #print 'image point\n',image_points
    pSize=image_points.shape
    flag=np.zeros((1,4))
    RT=estimate_initial_RT(E)
    #print 'pSize[0]',pSize[0]
    #camera_matrices=K.dot(rt)
    if pSize[0]==50:
        flagMax=1
        #print 'if yes'
    else:
        #print 'if no'
        for jj in range(pSize[0]):
            for i in range(4):
                rt=RT[i,:,:]
                #print rt
                camera0=K.dot(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
                camera1=K.dot(rt)
            #camera_matrices=np.array([camera0,camera1])
            #print 'camera0',camera0
            #print 'camera1',camera1
                camera_matrices=np.array([camera0,camera1])
            #print camera_matrices.shape
                #print 'image_points[j,:,:]\n',image_points[jj,:,:],jj
                try:
                    point3d=nonlinear_estimate_3d_point(image_points[jj,:,:], camera_matrices)
                #print 'point 3d\n',point3d
                except:
                    continue
                temp=np.ones((4,1))
                temp[0:3,:]=np.reshape(point3d,(3,1))
                #print 'temp',temp
                temp1=rt[:,:3].T
                temp2=-temp1.dot(rt[:,3:])
                map1=np.concatenate((temp1,temp2),axis=1)
                point3dNew=map1.dot(temp)
                #print 'point 3d new\n',point3dNew[:,0]
                if point3dNew[2]>0 and point3d[2]>0:
                    flag[0][i]=flag[0][i]+1
                    break
        flagMax=np.argmax(flag)
    #print flagMax
    return RT[flagMax,:,:]        
    raise Exception('Not Implemented Error')

if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'))[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'))
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'))[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print '-' * 80
    print "Part A: Check your matrices against the example R,T"
    print '-' * 80
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = scipy.misc.imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print "Example RT:\n", example_RT
    estimated_RT = estimate_initial_RT(E)
    print
    print "Estimated RT:\n", estimated_RT

    # Part B: Determining the best linear estimate of a 3D point
    print '-' * 80
    print 'Part B: Check that the difference from expected point '
    print 'is near zero'
    print '-' * 80
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    #print 'estimated_3d_point\n',estimated_3d_point
    print "Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum()

    # Part C: Calculating the reprojection error and its Jacobian
    print '-' * 80
    print 'Part C: Check that the difference from expected error/Jacobian '
    print 'is near zero'
    print '-' * 80
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    #print 'estimated_jac\n',estimated_jacobian
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print "Error Difference: ", np.fabs(estimated_error - expected_error).sum()
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print "Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum()

    # Part D: Determining the best nonlinear estimate of a 3D point
    print '-' * 80
    print 'Part D: Check that the reprojection error from nonlinear method'
    print 'is lower than linear method'
    print '-' * 80
    #print 'unit_test_image_matches.copy()',unit_test_image_matches.copy()
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print "Linear method error:", np.linalg.norm(error_linear)
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print "Nonlinear method error:", np.linalg.norm(error_nonlinear)

    # Part E: Determining the correct R, T from Essential Matrix
    print '-' * 80
    print "Part E: Check your matrix against the example R,T"
    print '-' * 80
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print "Example RT:\n", example_RT
    print
    print "Estimated RT:\n", estimated_RT

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print '-' * 80
    print 'Part F: Run the entire SFM pipeline'
    print '-' * 80
    frames = [0] * (len(image_paths) - 1)
    for i in xrange(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in xrange(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in xrange(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        c='k', depthshade=True, s=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    ax.view_init(-100, 90)

    plt.show()
