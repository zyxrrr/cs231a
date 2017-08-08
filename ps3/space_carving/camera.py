import numpy as np

class Camera:

    def __init__(self, K, P, true_silhouette):
        pass

    def __init__(self, frame):
        self.image = frame[0]
        self.P = frame[1]
        self.K = frame[2]
        self.R = frame[3]
        self.T = frame[4][:,0]
        self.silhouette = frame[6]

    # Get the unit vector for the direction of the camera
    def get_camera_direction(self):
        x = np.array([self.image.shape[1] / 2,
             self.image.shape[0] / 2,
             1]);
        X = np.linalg.solve(self.K, x)
        X = self.R.transpose().dot(X)
        return X / np.linalg.norm(X)
