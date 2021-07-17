import math
from collections import deque

import numpy as np
from numba import njit

@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.
    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.
        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)
    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def walk_along_trajectory(trajectory, t, i, distance):
    iplus = i + 1
    if iplus == trajectory.shape[0]:
        iplus = 0
    acc = -t * np.linalg.norm(trajectory[iplus, :] - trajectory[i, :])
    while acc <= distance:
        i += 1
        if i == trajectory.shape[0]:
            i = 0
        acc += np.linalg.norm(trajectory[i, :] - trajectory[i - 1, :])
    lenlast = np.linalg.norm(trajectory[i, :] - trajectory[i - 1, :])
    t_out = 1 - ((acc - distance) / lenlast)
    point_out = trajectory[i - 1, :] + t_out * (trajectory[i, :] - trajectory[i - 1, :])
    i_out = i - 1
    if i_out == -1:
        i_out = trajectory.shape[0]
    return point_out, t_out, i_out

waypoints = np.loadtxt("waypoints.csv", delimiter=',', dtype=np.float32)
print(waypoints.shape)
point = np.array([0, 0.51], dtype=np.float32)
nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(point, waypoints)
print(t, i)
print(walk_along_trajectory(waypoints, 0.5, 0, 1))
