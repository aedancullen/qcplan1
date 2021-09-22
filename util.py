import numpy as np
from numba import njit

@njit(cache=True)
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

@njit(cache=True)
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
    diff = trajectory[i, :] - trajectory[i - 1, :]
    point_out = trajectory[i - 1, :] + t_out * diff
    i_out = i - 1
    if i_out == -1:
        i_out = trajectory.shape[0]

    angle_out = np.arctan2(diff[1], diff[0])
    return point_out, angle_out, t_out, i_out

@njit(cache=True)
def discretize(arrsize, subdiv, value):
    return round(value * subdiv + arrsize / 2)

@njit(cache=True)
def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
    """

    # positive accl limit
    if vel > v_switch:
        pos_limit = a_max*v_switch/vel
    else:
        pos_limit = a_max

    # accl limit reached?
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit

    return accl

@njit(cache=True)
def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max

    return steering_velocity

@njit(cache=True)
def vehicle_dynamics_ks(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Kinematic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lwb = lf + lr

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # system dynamics
    f = np.array([x[3]*np.cos(x[4]),
         x[3]*np.sin(x[4]),
         u[0],
         u[1],
         x[3]/lwb*np.tan(x[2])])
    return f

@njit(cache=True)
def vehicle_dynamics_st(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Dynamic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
                x6: yaw rate
                x7: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """

    # gravity constant m/s^2
    g = 9.81

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.5:
        # wheelbase
        lwb = lf + lr

        # system dynamics
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
        f = np.hstack((f_ks, np.array([u[1]/lwb*np.tan(x[2])+x[3]/(lwb*np.cos(x[2])**2)*u[0],
        0])))

    else:
        # system dynamics
        f = np.array([x[3]*np.cos(x[6] + x[4]),
            x[3]*np.sin(x[6] + x[4]),
            u[0],
            u[1],
            x[5],
            -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
                +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
                +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
            (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
                -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
                +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])

    return f

@njit(cache=True)
def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    """
    Basic controller for speed/steer -> accl./steer vel.

        Args:
            speed (float): desired input speed
            steer (float): desired input steering angle

        Returns:
            accl (float): desired input acceleration
            sv (float): desired input steering velocity
    """
    # steering
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        sv = (steer_diff / np.fabs(steer_diff)) * max_sv
    else:
        sv = 0.0

    # accl
    vel_diff = speed - current_speed
    # currently forward
    if current_speed > 0.:
        if (vel_diff > 0):
            # accelerate
            kp = 2.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # braking
            kp = 2.0 * max_a / (-min_v)
            accl = kp * vel_diff
    # currently backwards
    else:
        if (vel_diff > 0):
            # braking
            kp = 2.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # accelerating
            kp = 2.0 * max_a / (-min_v)
            accl = kp * vel_diff

    return accl, sv

@njit(cache=True)
def rangefind(np_state, latched_map, map_subdiv, direction, max_dist, width):
    step = 1 / map_subdiv / 2
    cos_step = np.cos(direction) * step
    sin_step = np.sin(direction) * step
    cos_step_perp = np.cos(direction + np.pi / 2) * step
    sin_step_perp = np.sin(direction + np.pi / 2) * step
    trans_x = np_state[0]
    trans_y = np_state[1]
    dist = 0
    while dist < max_dist:
        trans_x += cos_step
        trans_y += sin_step
        dist += step
        trans_x_perp = 0
        trans_y_perp = 0
        dist_perp = 0
        while dist_perp < width / 2:
            trans_x_perp += cos_step_perp
            trans_y_perp += sin_step_perp
            dist_perp += step
            x1 = discretize(latched_map.shape[0], map_subdiv, trans_x + trans_x_perp);
            y1 = discretize(latched_map.shape[1], map_subdiv, trans_y + trans_y_perp);
            x2 = discretize(latched_map.shape[0], map_subdiv, trans_x - trans_x_perp);
            y2 = discretize(latched_map.shape[1], map_subdiv, trans_y - trans_y_perp);
            if latched_map[x1, y1] >= 128 or latched_map[x2, y2] >= 128:
                return dist
    return max_dist

@njit(cache=True)
def tangent_bug(np_state, latched_map, map_subdiv, goal_point, goal_angle, direction_step, cont_thresh, width):
    goal_direction = np.arctan2(goal_point[1] - np_state[1], goal_point[0] - np_state[0])
    goal_dist = np.linalg.norm(goal_point - np_state[:2])
    dist = rangefind(np_state, latched_map, map_subdiv, goal_direction, goal_dist, width)
    target = goal_direction
    if dist < goal_dist:
        l_last = dist
        r_last = dist
        sweep_direction_l = goal_direction
        sweep_direction_r = goal_direction
        for i in range(round((np.pi / 2) / direction_step)):
            sweep_direction_l += direction_step
            sweep_direction_r -= direction_step
            l_new = rangefind(np_state, latched_map, map_subdiv, sweep_direction_l, goal_dist * 2, width)
            r_new = rangefind(np_state, latched_map, map_subdiv, sweep_direction_r, goal_dist * 2, width)
            if l_new - l_last > cont_thresh: # went farther
                target = sweep_direction_l + direction_step
                break
            if l_last - l_new > cont_thresh: # came nearer; backtrack
                target = sweep_direction_l - 2 * direction_step
                break
            if r_new - r_last > cont_thresh: # went farther
                target = sweep_direction_r - direction_step
                break
            if r_last - r_new > cont_thresh: # came nearer; backtrack
                target = sweep_direction_r + 2 * direction_step
                break
            l_last = l_new
            r_last = r_new
        #if i == round((np.pi / 2) / direction_step) - 1:
            #print("oof")

    target_aim = target - np_state[2]
    if target_aim < -np.pi:
        target_aim += 2*np.pi
    elif target_aim >= np.pi:
        target_aim -= 2*np.pi

    goal_direction_aim = goal_direction - np_state[2]
    if goal_direction_aim < -np.pi:
        goal_direction_aim += 2*np.pi
    elif goal_direction_aim >= np.pi:
        goal_direction_aim -= 2*np.pi

    return target_aim, goal_direction_aim

@njit(cache=True)
def farthest_target(np_state, latched_map, map_subdiv, goal_point, goal_angle, direction_step, width):
    front_dist = rangefind(np_state, latched_map, map_subdiv, np_state[2], 100, width)

    goal_direction = np.arctan2(goal_point[1] - np_state[1], goal_point[0] - np_state[0])
    goal_dist = np.linalg.norm(goal_point - np_state[:2])
    dist = rangefind(np_state, latched_map, map_subdiv, goal_direction, goal_dist, width)
    if dist >= goal_dist:
        target_aim = goal_direction - np_state[2]
        if target_aim < -np.pi:
            target_aim += 2*np.pi
        elif target_aim >= np.pi:
            target_aim -= 2*np.pi
        return target_aim, front_dist

    sweep_direction = -np.pi / 2
    target_aim = sweep_direction
    target_dist = 0
    for i in range(round(np.pi / direction_step)):
        sweep_direction += direction_step
        dist = rangefind(np_state, latched_map, map_subdiv, np_state[2] + sweep_direction, 100, width)
        if dist > target_dist:
            target_dist = dist
            target_aim = sweep_direction

    return target_aim, front_dist

@njit(cache=True)
def fast_state_validity_check(np_state, latched_map, map_subdiv, length, width):
    direction = np_state[2]
    step = 1 / map_subdiv / 2
    cos_step = np.cos(direction) * step
    sin_step = np.sin(direction) * step
    cos_step_perp = np.cos(direction + np.pi / 2) * step
    sin_step_perp = np.sin(direction + np.pi / 2) * step
    trans_x = np_state[0] - np.cos(direction) * length / 2
    trans_y = np_state[1] - np.sin(direction) * length / 2
    dist = 0
    while dist < length:
        trans_x += cos_step
        trans_y += sin_step
        dist += step
        trans_x_perp = 0
        trans_y_perp = 0
        dist_perp = 0
        while dist_perp < width / 2:
            trans_x_perp += cos_step_perp
            trans_y_perp += sin_step_perp
            dist_perp += step
            x1 = discretize(latched_map.shape[0], map_subdiv, trans_x + trans_x_perp);
            y1 = discretize(latched_map.shape[1], map_subdiv, trans_y + trans_y_perp);
            x2 = discretize(latched_map.shape[0], map_subdiv, trans_x - trans_x_perp);
            y2 = discretize(latched_map.shape[1], map_subdiv, trans_y - trans_y_perp);
            if latched_map[x1, y1] >= 128 or latched_map[x2, y2] >= 128:
                return False
    return True

@njit(cache=True)
def fast_state_propagate(np_state, steer0, steer, control, duration, physics_timestep,
                         mu,
                         C_Sf,
                         C_Sr,
                         lf,
                         lr,
                         h,
                         m,
                         I,
                         s_min,
                         s_max,
                         sv_min,
                         sv_max,
                         v_switch,
                         a_max,
                         v_min,
                         v_max):

    vel = control[1]

    for i in range(int(duration)):
        # bound yaw angle
        if np_state[4] > 2*np.pi:
            np_state[4] -= 2*np.pi
        elif np_state[4] < 0:
            np_state[4] += 2*np.pi

        # steering angle velocity input to steering velocity acceleration input
        accl, sv = pid(vel, steer, np_state[3], np_state[2], sv_max, a_max, v_max, v_min)

        # update physics, get RHS of diff'eq
        f = vehicle_dynamics_st(
            np_state,
            np.array([sv, accl]),
            mu,
            C_Sf,
            C_Sr,
            lf,
            lr,
            h,
            m,
            I,
            s_min,
            s_max,
            sv_min,
            sv_max,
            v_switch,
            a_max,
            v_min,
            v_max)

        # update state
        np_state = np_state + f * physics_timestep

        steer = steer0
        steer0 = control[0]

    if np_state[4] < -np.pi:
        np_state[4] += 2*np.pi
    elif np_state[4] >= np.pi:
        np_state[4] -= 2*np.pi

    return np_state, steer0, steer

@njit(cache=True)
def combine_scan(np_state, latched_map, map_subdiv, ranges, angle_min, angle_increment):
    for i in range(len(ranges)):
        dist = ranges[i]
        angle = np_state[2] + angle_min + i * angle_increment
        location_x = np_state[0] + dist * np.cos(angle)
        location_y = np_state[1] + dist * np.sin(angle)
        bi_x = discretize(latched_map.shape[0], map_subdiv, location_x)
        bi_y = discretize(latched_map.shape[1], map_subdiv, location_y)
        latched_map[bi_x, bi_y] = 255
