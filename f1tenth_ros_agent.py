#!/usr/bin/python3
import os

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from f1tenth_gym_ros.msg import RaceInfo
from tf.transformations import euler_from_quaternion

import numpy as np
from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

import util

#ou.setLogLevel(ou.LOG_WARN)

PARAMS = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

NUM_CONTROLS = 2
CONTROL_LOWER = [PARAMS["s_min"], PARAMS["v_min"]]
CONTROL_UPPER = [PARAMS["s_max"], PARAMS["v_max"]]

GRIDMAP_XY_SUBDIV = 1/0.07712

PHYSICS_TIMESTEP = 0.01 # Actual value used in calculation
SIM_INTERVAL = 0.02 # Real time interval of simulator's internal physics callbacks

CHUNK_MULTIPLIER = 10

CHUNK_DURATION = SIM_INTERVAL * CHUNK_MULTIPLIER
CHUNK_DISTANCE = 3
GOAL_THRESHOLD = 2

class QCPassControlSampler(oc.ControlSampler):
    def __init__(self, controlspace, latched_map, goal_state):
        super().__init__(controlspace)
        self.latched_map = latched_map
        self.goal_state = goal_state

    def sample(self, control, start_state):
        #for i in range(NUM_CONTROLS):
        #    control[i] = np.random.uniform(CONTROL_LOWER[i], CONTROL_UPPER[i])

        goalx = self.goal_state[0].getX()
        goaly = self.goal_state[0].getY()
        startx = start_state[0].getX()
        starty = start_state[0].getY()
        c0mean = np.arctan2(goaly - starty, goalx - startx) - start_state[0].getYaw()
        if c0mean < -np.pi:
            c0mean += 2.0 * np.pi
        elif c0mean >= np.pi:
            c0mean -= 2.0 * np.pi
        control[0] = np.clip(np.random.normal(c0mean, 0.1), CONTROL_LOWER[0], CONTROL_UPPER[0])
        control[1] = np.random.uniform(CONTROL_LOWER[1], CONTROL_UPPER[1])

class QCPlan1:
    def __init__(self, hardware_map, waypoints_fn, gridmap_fn):
        self.hardware_map = hardware_map

        self.gridmap = np.load(gridmap_fn)
        self.waypoints = np.loadtxt(waypoints_fn, delimiter=',', dtype=np.float32)
        self.waypoints[:, 0] -= self.gridmap.shape[0] / 2
        self.waypoints[:, 1] -= self.gridmap.shape[1] / 2
        self.waypoints /= GRIDMAP_XY_SUBDIV

        self.se2space = ob.SE2StateSpace()
        #self.se2space.setSubspaceWeight(0, 1) # R^2 subspace weight 1
        #self.se2space.setSubspaceWeight(1, 0) # SO(2) subspace weight 0
        self.vectorspace = ob.RealVectorStateSpace(6)
        self.vectorbounds = ob.RealVectorBounds(6)
        self.vectorbounds.setLow(-99999) # don't care
        self.vectorbounds.setHigh(99999)
        self.vectorspace.setBounds(self.vectorbounds)

        self.statespace = ob.CompoundStateSpace()
        self.statespace.addSubspace(self.se2space, 1) # weight 1
        self.statespace.addSubspace(self.vectorspace, 0) # weight 0

        self.controlspace = oc.RealVectorControlSpace(self.statespace, 2)
        self.controlspace.setControlSamplerAllocator(oc.ControlSamplerAllocator(self.csampler_alloc))

        self.ss = oc.SimpleSetup(self.controlspace)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.state_validity_check))
        self.ss.setStatePropagator(oc.StatePropagatorFn(self.state_propagate))

        self.si = self.ss.getSpaceInformation()
        self.si.setPropagationStepSize(CHUNK_MULTIPLIER)
        self.si.setMinMaxControlDuration(1, 1)

        #========

        self.last_physics_ticks_elapsed = 0
        self.last_control = [0, 0]
        self.control = None

        print("====>", "Waiting for hardware map")
        while not self.hardware_map.ready():
            rospy.sleep(0.001)
        
        self.state = ob.State(self.statespace)
        state = self.state()
        if self.hardware_map.scan.header.frame_id.startswith("ego"):
            i = 0
            print("====>", "Identity is ego")
        else:
            i = 1
            print("====>", "Identity is opp")
        state[0].setX(0. + (i * 0.75))
        state[0].setY(0. - (i*1.5))
        state[0].setYaw(np.radians(60))
        state[1][0] = 0
        state[1][1] = 0
        state[1][2] = 0
        state[1][3] = 0
        state[1][4] = 0
        state[1][5] = 0

    def loop(self, timer):
        if self.control is not None:
            self.hardware_map.drive(self.control[0], self.control[1])

        odom_captured = self.hardware_map.odom
        scan_captured = self.hardware_map.scan

        # Update real state
        physics_ticks_elapsed = round(self.hardware_map.race_info.ego_elapsed_time / PHYSICS_TIMESTEP)
        physics_ticks_new = physics_ticks_elapsed - self.last_physics_ticks_elapsed
        self.last_physics_ticks_elapsed = physics_ticks_elapsed
        self.state_propagate(self.state(), self.last_control, physics_ticks_new, self.state())
        self.state()[0].setX(odom_captured.pose.pose.position.x)
        self.state()[0].setY(odom_captured.pose.pose.position.y)
        x, y, z = euler_from_quaternion([
            odom_captured.pose.pose.orientation.x,
            odom_captured.pose.pose.orientation.y,
            odom_captured.pose.pose.orientation.z,
            odom_captured.pose.pose.orientation.w,
        ])
        self.state()[0].setYaw(z)
        self.state()[1][1] = odom_captured.twist.twist.linear.x
        self.state()[1][2] = odom_captured.twist.twist.angular.z
        self.statespace.enforceBounds(self.state())

        # Latch map
        np_state = np.array([self.state()[0].getX(), self.state()[0].getY(), self.state()[0].getYaw()])
        self.latched_map = self.gridmap.copy()
        util.combine_scan(
            np_state,
            self.latched_map,
            GRIDMAP_XY_SUBDIV,
            np.array(scan_captured.ranges),
            scan_captured.angle_min,
            scan_captured.angle_increment,
        )

        # Predict future state if controls were issued
        future_state = ob.State(self.statespace)
        if self.control is not None:
            self.state_propagate(self.state(), self.control, CHUNK_MULTIPLIER, future_state())
            self.last_control = self.control
        else:
            self.statespace.copyState(future_state(), self.state())

        # Plan from future state
        self.ss.clear()
        self.ss.setStartState(future_state)
        start_point = np.array([future_state()[0].getX(), future_state()[0].getY()], dtype=np.float32)
        nearest_point, nearest_dist, t, i = util.nearest_point_on_trajectory(start_point, self.waypoints)
        goal_point, goal_angle, t, i = util.walk_along_trajectory(self.waypoints, t, i, CHUNK_DISTANCE)
        self.goal_state = ob.State(self.statespace)
        self.goal_state()[0].setX(goal_point[0])
        self.goal_state()[0].setY(goal_point[1])
        self.goal_state()[0].setYaw(goal_angle)
        self.ss.setGoalState(self.goal_state, GOAL_THRESHOLD)
        self.se2bounds = ob.RealVectorBounds(2)
        self.se2bounds.setLow(0, min(goal_point[0], start_point[0]) - CHUNK_DISTANCE)
        self.se2bounds.setLow(1, min(goal_point[1], start_point[1]) - CHUNK_DISTANCE)
        self.se2bounds.setHigh(0, max(goal_point[0], start_point[0]) + CHUNK_DISTANCE)
        self.se2bounds.setHigh(1, max(goal_point[1], start_point[1]) + CHUNK_DISTANCE)
        self.se2space.setBounds(self.se2bounds)
        self.planner = oc.SST(self.si)
        #self.planner.setPruningRadius(0.01) # tenth of default
        #self.planner.setSelectionRadius(0.02) # tenth of default
        self.planner.setPruningRadius(0.00)
        self.ss.setPlanner(self.planner)
        solved = self.ss.solve(CHUNK_DURATION - 0.010)
        if solved:
            solution = self.ss.getSolutionPath()
            controls = solution.getControls()
            count = solution.getControlCount()
            if self.ss.haveExactSolutionPath():
                self.control = [controls[0][0], controls[0][1]]
                print("====> Complete:", count, self.control)
            else:
                self.control = [controls[0][0], 0]
                print("====> Incomplete:", count, self.control)
        else:
            self.control = [0, 0]
            print("====>", "Not solved")

    def state_validity_check(self, state):
        np_state = np.array([state[0].getX(), state[0].getY(), state[0].getYaw()])
        return util.fast_state_validity_check(
            np_state,
            self.latched_map,
            GRIDMAP_XY_SUBDIV,
            PARAMS["length"],
            PARAMS["width"],
        )

    def state_propagate(self, start, control, duration, state):
        np_state = np.array([
            start[0].getX(),
            start[0].getY(),
            start[1][0],
            start[1][1],
            start[0].getYaw(),
            start[1][2],
            start[1][3],
        ])

        steer0 = start[1][4]
        steer = start[1][5]

        control = np.array([control[0], control[1]])

        np_state, steer0, steer = util.fast_state_propagate(
            np_state,
            steer0,
            steer,
            control,
            duration,
            PHYSICS_TIMESTEP,
            PARAMS['mu'],
            PARAMS['C_Sf'],
            PARAMS['C_Sr'],
            PARAMS['lf'],
            PARAMS['lr'],
            PARAMS['h'],
            PARAMS['m'],
            PARAMS['I'],
            PARAMS['s_min'],
            PARAMS['s_max'],
            PARAMS['sv_min'],
            PARAMS['sv_max'],
            PARAMS['v_switch'],
            PARAMS['a_max'],
            PARAMS['v_min'],
            PARAMS['v_max'],
        )

        state[0].setX(np_state[0])
        state[0].setY(np_state[1])
        state[0].setYaw(np_state[4])
        state[1][0] = np_state[2]
        state[1][1] = np_state[3]
        state[1][2] = np_state[5]
        state[1][3] = np_state[6]
        state[1][4] = steer0
        state[1][5] = steer

        self.statespace.enforceBounds(state)

    def csampler_alloc(self, control_space):
        return QCPassControlSampler(control_space, self.latched_map, self.goal_state())

class HardwareMap:
    def __init__(self):
        self.odom = None
        self.scan = None
        self.race_info = None

        self.odom_sub = rospy.Subscriber("/%s/odom" % agent_name, Odometry, self.odom_callback, queue_size=1)
        self.scan_sub = rospy.Subscriber("/%s/scan" % agent_name, LaserScan, self.scan_callback, queue_size=1)
        self.race_info_sub = rospy.Subscriber("/race_info", RaceInfo, self.race_info_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/%s/drive" % agent_name, AckermannDriveStamped, queue_size=1)

    def odom_callback(self, odom):
        self.odom = odom

    def scan_callback(self, scan):
        self.scan = scan

    def race_info_callback(self, race_info):
        self.race_info = race_info
        
    def drive(self, steering_angle, speed):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering_angle
        msg.drive.speed = speed
        self.drive_pub.publish(msg)
        
    def ready(self):
        return self.scan is not None and self.race_info is not None and self.odom is not None

if __name__ == "__main__":
    agent_name = os.environ.get("F1TENTH_AGENT_NAME")
    rospy.init_node("gym_agent_%s" % agent_name, anonymous=True)
    
    filepath = os.path.abspath(os.path.dirname(__file__))
    qc = QCPlan1(HardwareMap(),
        "%s/waypoints.csv" % filepath,
        "%s/gridmap.npy" % filepath,
    )
    loop_timer = rospy.Timer(rospy.Duration(CHUNK_DURATION), qc.loop)
    rospy.spin()
