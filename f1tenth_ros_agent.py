#!/usr/bin/python3
import os
import sys
import time

import rospy
from f1tenth_gym_ros.msg import RaceInfo
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance
from pkg.msg import Observation
from tf.transformations import euler_from_quaternion

import numpy as np
from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

import util

ou.setLogLevel(ou.LOG_ERROR)

PARAMS = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}#'width': 0.5, 'length': 0.8}#

NUM_CONTROLS = 2
CONTROL_LOWER = [PARAMS["s_min"], PARAMS["v_min"]]
CONTROL_UPPER = [PARAMS["s_max"], PARAMS["v_max"]]

GRIDMAP_XY_SUBDIV = 1/0.08534

PHYSICS_TIMESTEP = 0.01 # Actual value used in calculation
SIM_INTERVAL = 0.02 # Real time interval of simulator's internal physics callbacks

CHUNK_MULTIPLIER = 5

CHUNK_DURATION = SIM_INTERVAL * CHUNK_MULTIPLIER
CHUNK_DISTANCE = 7
GOAL_THRESHOLD = 2

HEURISTIC_DIRECTION_STEP = np.radians(0.1)
HEURISTIC_CONT_THRESH = 1
STEER_GAIN = 0.3
STEER_STDEV = 0.1
VEL_GAIN = 2
VEL_STDEV = 10

class QCPassControlSampler(oc.ControlSampler):
    def __init__(self, controlspace, latched_map, goal_point, goal_angle):
        super().__init__(controlspace)
        self.latched_map = latched_map
        self.goal_point = goal_point
        self.goal_angle = goal_angle

    def sample(self, control, state, selections):
        np_state = np.array([state[0].getX(), state[0].getY(), state[0].getYaw()])

        target = util.tangent_bug(
            np_state,
            self.latched_map,
            GRIDMAP_XY_SUBDIV,
            self.goal_point,
            HEURISTIC_DIRECTION_STEP,
            HEURISTIC_CONT_THRESH,
            PARAMS["width"],
        )
        front_dist = util.rangefind(np_state, self.latched_map, GRIDMAP_XY_SUBDIV, np_state[2], 100)

        if selections == 0:
            c0 = np.clip(target * STEER_GAIN, CONTROL_LOWER[0], CONTROL_UPPER[0])
            c1 = np.clip(front_dist * VEL_GAIN, CONTROL_LOWER[1], CONTROL_UPPER[1])
        else:
            c0 = np.random.normal(np.clip(target * STEER_GAIN, CONTROL_LOWER[0], CONTROL_UPPER[0]), STEER_STDEV)
            c1 = np.random.normal(np.clip(front_dist * VEL_GAIN, CONTROL_LOWER[1], CONTROL_UPPER[1]), VEL_STDEV)

        control[0] = np.clip(c0, CONTROL_LOWER[0], CONTROL_UPPER[0])
        control[1] = np.clip(c1, CONTROL_LOWER[1], CONTROL_UPPER[1])

class QCPlan1:
    def __init__(self, hardware_map, waypoints_fn, gridmap_fn):
        self.hardware_map = hardware_map

        self.gridmap = np.load(gridmap_fn)
        self.waypoints = np.loadtxt(waypoints_fn, delimiter=',')
        self.waypoints[:, 0] -= self.gridmap.shape[0] / 2
        self.waypoints[:, 1] -= self.gridmap.shape[1] / 2
        self.waypoints /= GRIDMAP_XY_SUBDIV

        self.se2space = ob.SE2StateSpace()
        self.se2bounds = ob.RealVectorBounds(2)
        self.se2bounds.setLow(-99999) # don't care
        self.se2bounds.setHigh(99999)
        self.se2space.setBounds(self.se2bounds)

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
        self.si.setPropagationStepSize(1)
        self.si.setMinMaxControlDuration(CHUNK_MULTIPLIER, CHUNK_MULTIPLIER)

        #========

        self.last_control = [0, 0]
        self.control = None

        print("====>", "Waiting for hardware map")
        while not self.hardware_map.ready():
            rospy.sleep(0.001)

        self.state = ob.State(self.statespace)
        state = self.state()
        state[1][0] = 0
        state[1][1] = 0
        state[1][2] = 0
        state[1][3] = 0
        state[1][4] = 0
        state[1][5] = 0
        self.se2space.setBounds(self.se2bounds)
        self.statespace.enforceBounds(self.state())

    def loop(self, timer):
        start = time.time()

        if self.control is not None:
            self.hardware_map.drive(self.control[0], self.control[1])

        obs_captured = self.hardware_map.observations

        # Update real state
        self.state_propagate(self.state(), self.last_control, CHUNK_MULTIPLIER, self.state())
        self.state()[0].setX(obs_captured.ego_pose.pose.position.x)
        self.state()[0].setY(obs_captured.ego_pose.pose.position.y)
        x, y, z = euler_from_quaternion([
            obs_captured.ego_pose.pose.orientation.x,
            obs_captured.ego_pose.pose.orientation.y,
            obs_captured.ego_pose.pose.orientation.z,
            obs_captured.ego_pose.pose.orientation.w,
        ])
        self.state()[0].setYaw(z)
        self.state()[1][1] = obs_captured.ego_twist.twist.linear.x
        self.state()[1][2] = obs_captured.ego_twist.twist.angular.z
        self.se2space.setBounds(self.se2bounds)
        self.statespace.enforceBounds(self.state())

        # Latch map
        np_state = np.array([self.state()[0].getX(), self.state()[0].getY(), self.state()[0].getYaw()])
        self.latched_map = self.gridmap.copy()
        util.combine_scan(
            np_state,
            self.latched_map,
            GRIDMAP_XY_SUBDIV,
            np.array(obs_captured.ranges),
            self.hardware_map.angle_min,
            self.hardware_map.angle_inc,
        )

        # Predict future state if controls were issued
        future_state = ob.State(self.statespace)
        if self.control is not None:
            self.state_propagate(self.state(), self.control, CHUNK_MULTIPLIER, future_state())
            self.last_control = self.control
        else:
            self.statespace.copyState(future_state(), self.state())

        # Plan from future state
        self.planner = oc.SST(self.si)
        self.planner.setPruningRadius(0.00)
        self.planner.setSelectionRadius(0.00)
        if self.ss.getLastPlannerStatus():
            # Copy old path into a new PathControl because it will be freed on self.ss.clear()
            seed_path = oc.PathControl(self.ss.getSolutionPath())
            self.planner.setSeedPath(seed_path, 1)

        self.ss.clear()
        self.ss.setStartState(future_state)
        start_point = np.array([future_state()[0].getX(), future_state()[0].getY()])
        nearest_point, nearest_dist, t, i = util.nearest_point_on_trajectory(start_point, self.waypoints)
        self.goal_point, self.goal_angle, t, i = util.walk_along_trajectory(self.waypoints, t, i, CHUNK_DISTANCE)

        goal_state = ob.State(self.statespace)
        goal_state()[0].setX(self.goal_point[0])
        goal_state()[0].setY(self.goal_point[1])
        goal_state()[0].setYaw(self.goal_angle)
        self.ss.setGoalState(goal_state, GOAL_THRESHOLD)
        planbounds = ob.RealVectorBounds(2)
        planbounds.setLow(0, min(self.goal_point[0], start_point[0]) - CHUNK_DISTANCE / 2)
        planbounds.setLow(1, min(self.goal_point[1], start_point[1]) - CHUNK_DISTANCE / 2)
        planbounds.setHigh(0, max(self.goal_point[0], start_point[0]) + CHUNK_DISTANCE / 2)
        planbounds.setHigh(1, max(self.goal_point[1], start_point[1]) + CHUNK_DISTANCE / 2)
        self.se2space.setBounds(planbounds)

        self.ss.setPlanner(self.planner)
        solved = self.ss.solve(CHUNK_DURATION - 0.010)
        print("====>", round(self.state()[1][1]), "m/s,", round((time.time() - start) * 1000), "ms, ", end='')
        if solved:
            solution = self.ss.getSolutionPath()
            controls = solution.getControls()
            count = solution.getControlCount()
            if self.ss.haveExactSolutionPath():
                self.control = [controls[0][0], controls[0][1]]
                print("complete:", count, "segments, c1 =", round(self.control[1]))
            else:
                self.control = [controls[0][0], controls[0][1]]
                print("incomplete:", count, "segments, c1 =", round(self.control[1]))
        else:
            print("not solved")

    def state_validity_check(self, state):
        np_state = np.array([state[0].getX(), state[0].getY(), state[0].getYaw()])
        return util.fast_state_validity_check(
            np_state,
            self.latched_map,
            GRIDMAP_XY_SUBDIV,
            PARAMS["length"],
            PARAMS["width"],
        ) and self.statespace.satisfiesBounds(state)

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
            PARAMS["mu"],
            PARAMS["C_Sf"],
            PARAMS["C_Sr"],
            PARAMS["lf"],
            PARAMS["lr"],
            PARAMS["h"],
            PARAMS["m"],
            PARAMS["I"],
            PARAMS["s_min"],
            PARAMS["s_max"],
            PARAMS["sv_min"],
            PARAMS["sv_max"],
            PARAMS["v_switch"],
            PARAMS["a_max"],
            PARAMS["v_min"],
            PARAMS["v_max"],
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

    def csampler_alloc(self, controlspace):
        return QCPassControlSampler(controlspace, self.latched_map, self.goal_point, self.goal_angle)

class HardwareMap:
    def __init__(self):
        self.observations = None

        scan_fov = rospy.get_param('scan_fov')
        scan_beams = rospy.get_param('scan_beams')
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / scan_beams

        self.observations_sub = rospy.Subscriber("/%s/observations" % agent_name, Observation, self.observations_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/%s/drive" % agent_name, AckermannDriveStamped, queue_size=1)

    def observations_callback(self, observations):
        self.observations = observations

    def drive(self, steering_angle, speed):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering_angle
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def ready(self):
        return self.observations is not None

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
