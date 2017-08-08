import numpy as np
import scipy.io as sio
import argparse
from camera import Camera
from plotting import *
import matplotlib.pyplot as plt


# A very simple, but useful method to take the difference between the
# first and second element (usually for 2D vectors)
def diff(x):
    return x[1] - x[0]


'''
FORM_INITIAL_VOXELS  create a basic grid of voxels ready for carving

Arguments:
    xlim - The limits of the x dimension given as [xmin xmax]

    ylim - The limits of the y dimension given as [ymin ymax]

    zlim - The limits of the z dimension given as [zmin zmax]

    num_voxels - The approximate number of voxels we desire in our grid

Returns:
    voxels - An ndarray of size (N, 3) where N is approximately equal the 
        num_voxels of voxel locations.

    voxel_size - The distance between the locations of adjacent voxels
        (a voxel is a cube)

Our initial voxels will create a rectangular prism defined by the x,y,z
limits. Each voxel will be a cube, so you'll have to compute the
approximate side-length (voxel_size) of these cubes, as well as how many
cubes you need to place in each dimension to get around the desired
number of voxel. This can be accomplished by first finding the total volume of
the voxel grid and dividing by the number of desired voxels. This will give an
approximate volume for each cubic voxel, which you can then use to find the 
side-length. The final "voxels" output should be a ndarray where every row is
the location of a voxel in 3D space.
'''
def form_initial_voxels(xlim, ylim, zlim, num_voxels):
    # TODO: Implement this method!
    #num_length=num_voxel_size**(1/3)
    #num_length=round(num_length)
    #num_voxels=num_length**(3)
    x_len=(xlim[1]-xlim[0])
    y_len=(ylim[1]-ylim[0])
    z_len=(zlim[1]-zlim[0])
    voxel_step3=x_len*y_len*z_len/(num_voxels)
    #print voxel_step3
    voxels_size=voxel_step3**(1./3)
    #print 'voxel_setp',voxels_size    
    x_num=int(x_len/voxels_size)
    y_num=int(y_len/voxels_size)
    z_num=int(z_len/voxels_size)
    voxels=np.zeros((x_num*y_num*z_num,3))
    #print x_num,y_num,z_num
    x_pos=range(x_num)
    x_pos=np.matlib.repmat(x_pos,y_num,1)
    x_size=x_pos.shape
    temp1=voxels_size*np.ones(x_size)
    x_pos=np.multiply(x_pos,temp1)+voxels_size/2
    x_pos=x_pos.reshape(x_size[0]*x_size[1],1)
    x_pos=np.matlib.repmat(x_pos,z_num,1)
    #print 'x_pos\n',x_pos
    #x_pos=x_pos[:]
    y_pos=range(y_num)
    y_pos=(np.matlib.repmat(y_pos,x_num,1)).T
    y_size=y_pos.shape
    temp1=voxels_size*np.ones(y_size)
    y_pos=np.multiply(y_pos,temp1)+voxels_size/2
    y_pos=y_pos.reshape(y_size[0]*y_size[1],1)
    y_pos=np.matlib.repmat(y_pos,z_num,1)
    #print 'y_pos\n',y_pos
    z_pos=range(z_num)
    z_pos=(np.matlib.repmat(z_pos,y_size[0]*y_size[1],1)).T
    z_size=z_pos.shape
    temp1=voxels_size*np.ones(z_size)
    z_pos=np.multiply(z_pos,temp1)+voxels_size/2
    zsize=z_pos.shape
    #print zsize
    z_pos=z_pos.reshape(zsize[0]*zsize[1],1)
    #print z_pos
    voxels[:,0:1]=x_pos
    voxels[:,1:2]=y_pos
    voxels[:,2:3]=z_pos
    return voxels,voxels_size
    raise Exception('Not Implemented Error')


'''
GET_VOXEL_BOUNDS: Gives a nice bounding box in which the object will be carved
from. We feed these x/y/z limits into the construction of the inital voxel
cuboid. 

Arguments:
    cameras - The given data, which stores all the information
        associated with each camera (P, image, silhouettes, etc.)

    estimate_better_bounds - a flag that simply tells us whether to set tighter
        bounds. We can carve based on the silhouette we use.

    num_voxels - If estimating a better bound, the number of voxels needed for
        a quick carving.

Returns:
    xlim - The limits of the x dimension given as [xmin xmax]

    ylim - The limits of the y dimension given as [ymin ymax]

    zlim - The limits of the z dimension given as [zmin zmax]

The current method is to simply use the camera locations as the bounds. In the
section underneath the TODO, please implement a method to find tigther bounds:
One such approach would be to do a quick carving of the object on a grid with 
very few voxels. From this coarse carving, we can determine tighter bounds. Of
course, these bounds may be too strict, so we should have a buffer of one 
voxel_size around the carved object. 
'''
def get_voxel_bounds(cameras, estimate_better_bounds = False, num_voxels = 4000):
    camera_positions = np.vstack([c.T for c in cameras])
    xlim = [camera_positions[:,0].min(), camera_positions[:,0].max()]
    ylim = [camera_positions[:,1].min(), camera_positions[:,1].max()]
    zlim = [camera_positions[:,2].min(), camera_positions[:,2].max()]

    # For the zlim we need to see where each camera is looking. 
    camera_range = 0.6 * np.sqrt(diff( xlim )**2 + diff( ylim )**2)
    for c in cameras:
        viewpoint = c.T - camera_range * c.get_camera_direction()
        zlim[0] = min( zlim[0], viewpoint[2] )
        zlim[1] = max( zlim[1], viewpoint[2] )

    # Move the limits in a bit since the object must be inside the circle
    xlim = xlim + diff(xlim) / 4 * np.array([1, -1])
    ylim = ylim + diff(ylim) / 4 * np.array([1, -1])

    if estimate_better_bounds:
        # TODO: Implement this method!
        print cameras[0]
        for item in cameras[0]:
            print item
        tempx=np.array([0,0])
        tempy=np.array([0,0])
        tempz=np.array([0,0])
        numcam=len(cameras)
        xyzlim=np.zeros((numcam,3))
        #print numcam        
        for i in range(numcam):
            voxels,voxels_size=form_initial_voxels(xlim, ylim, zlim, num_voxels)
            #print cameras[0]
            voxels=carve(voxels, cameras[i])
            xtemp=voxels[:,0]
            ytemp=voxels[:,1]
            ztemp=voxels[:,2]
            print 'tempx\n',voxels
            try:
                xlim1=np.array([min(xtemp), max(xtemp)])
                ylim1=np.array([min(ytemp), max(ytemp)])
                zlim1=np.array([min(ztemp), max(ztemp)])
            
                tempx=np.concatenate((tempx,xlim1),axis=0)
                tempy=np.concatenate((tempy,ylim1),axis=0)
                tempz=np.concatenate((tempz,zlim1),axis=0)
                print 'temp\n'
                print tempx
                print tempy
                print tempz
            except:
                asdf=1
        #print ytemp
        xlim=[min(tempx),max(tempx)]
        ylim=[min(tempx),max(tempy)]
        zlim=[min(tempz),max(tempz)]
        print xlim,ylim,zlim
        return xlim, ylim, zlim
    raise Exception('Not Implemented Error')
    

'''
CARVE: carves away voxels that are not inside the silhouette contained in 
    the view of the camera. The resulting voxel array is returned.

Arguments:
    voxels - an Nx3 matrix where each row is the location of a cubic voxel

    camera - The camera we are using to carve the voxels with. Useful data
        stored in here are the "silhouette" matrix, "image", and the
        projection matrix "P". 

Returns:
    voxels - a subset of the argument passed that are inside the silhouette
'''
def carve(voxels, camera):
    # TODO: Implement this method!
    sil=camera.silhouette
    #print 'sil\n',sil
    idx=np.nonzero(sil)
    #print 'idx\n',idx
    idx=np.vstack([idx[0],idx[1]])#row 0 is height,row 1 is col
    P=camera.P
    #print sil.dtype
    #print 'idx shape\n',idx,idx.shape   
    
    vox_size=voxels.shape
    
    temp=np.ones((vox_size[0],1))
    points_3d=np.concatenate((voxels,temp),axis=1)
    points_2d=P.dot(points_3d.T)
    temp2=points_2d[2:,:]
    temp2=np.matlib.repmat(temp2,3,1)
    points_2d=np.divide(points_2d,temp2)[:2,:]
    points_2d=points_2d.astype(int)#row 0 is x(row) ,row 1 is y(col)
    #print 'point 2d\n',points_2d

    idx_size=points_2d.shape
    #print 'idx size',idx_size
    flag=np.zeros((idx_size[1],1))
    idx_y=idx[0,:]
    max_y=max(idx_y)
    min_y=min(idx_y)
    idx_x=idx[1,:]
    max_x=max(idx_x)
    min_x=min(idx_x)
    flag1=points_2d[1,:]>min_y
    flag2=points_2d[1,:]<max_y
    #print flag1,flag2
    flag12=np.logical_and(flag1,flag2)
    flag34=np.logical_and(points_2d[0,:]>min_x,points_2d[0,:]<max_x)
    flag=np.logical_and(flag12,flag34)
    #print 'flag\n',flag,flag.shape

    idx_true=np.argwhere(flag==True)
    points_2d=points_2d.T
    pnew=points_2d[idx_true,:]
    ps_img=sil[pnew[:,:,1],pnew[:,:,0]]
    f1=ps_img!=0
    flag[idx_true]=f1
    v_now=voxels[flag,:]  

    
    
    return v_now            
    raise Exception('Not Implemented Error')


'''
ESTIMATE_SILHOUETTE: Uses a very naive and color-specific heuristic to generate
the silhouette of an object

Arguments:
    im - The image containing a known object. An ndarray of size (H, W, C).

Returns:
    silhouette - An ndarray of size (H, W), where each pixel location is 0 or 1.
        If the (i,j) value is 0, then that pixel location in the original image 
        does not correspond to the object. If the (i,j) value is 1, then that
        that pixel location in the original image does correspond to the object.
'''
def estimate_silhouette(im):
    return np.logical_and(im[:,:,0] > im[:,:,2], im[:,:,0] > im[:,:,1] )


if __name__ == '__main__':
    estimate_better_bounds = True
    use_true_silhouette = True
    frames = sio.loadmat('frames.mat')['frames'][0]
    cameras = [Camera(x) for x in frames]

    # Generate the silhouettes based on a color heuristic
    if not use_true_silhouette:
        for i, c in enumerate(cameras):
            c.true_silhouette = c.silhouette
            c.silhouette = estimate_silhouette(c.image)
            if i == 0:
                plt.figure()
                plt.subplot(121)
                plt.imshow(c.true_silhouette, cmap = 'gray')
                plt.title('True Silhouette')
                plt.subplot(122)
                plt.imshow(c.silhouette, cmap = 'gray')
                plt.title('Estimated Silhouette')
                plt.show()

    # Generate the voxel grid
    # You can reduce the number of voxels for faster debugging, but
    # make sure you use the full amount for your final solution
    num_voxels = 4e3
    xlim, ylim, zlim = get_voxel_bounds(cameras, estimate_better_bounds)

    # This part is simply to test forming the initial voxel grid
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, 4000)
    plot_surface(voxels)
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)

    # Test the initial carving
    voxels = carve(voxels, cameras[0])
    if use_true_silhouette:
        plot_surface(voxels)

    # Result after all carvings
    for c in cameras:
        voxels = carve(voxels, c)  
    plot_surface(voxels, voxel_size)
