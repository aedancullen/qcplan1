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
from util import PARAMS

#ou.setLogLevel(ou.LOG_WARN)

NUM_CONTROLS = 2
CONTROL_LOWER = [PARAMS["s_min"], PARAMS["v_min"]]
CONTROL_UPPER = [PARAMS["s_max"], PARAMS["v_max"]]

GRIDMAP_XY_SUBDIV = 1/0.07712

BIASMAP_XY_SUBDIV = 1
BIASMAP_YAW_SUBDIV = 10
BIASMAP_CONTROL_STDEV = [(CONTROL_UPPER[i] - CONTROL_LOWER[i]) / 5 for i in range(NUM_CONTROLS)]

PHYSICS_TIMESTEP = 0.01 # Actual value used in calculation
SIM_INTERVAL = 0.02 # Real time interval of simulator's internal physics callbacks

CHUNK_MULTIPLIER = 10

CHUNK_DURATION = SIM_INTERVAL * CHUNK_MULTIPLIER
CHUNK_DISTANCE = 5
GOAL_THRESHOLD = 2
MIN_PLAN_LENGTH = 3

class BiasmapControlSampler(oc.ControlSampler):
    def __init__(self, controlspace, biasmap, biasmap_valid):
        super().__init__(controlspace)
        self.controlspace = controlspace
        self.biasmap = biasmap
        self.biasmap_valid = biasmap_valid
        self.exact_flags = np.ones_like(biasmap, dtype=bool)

    def sample(self, control, start_state):
        bi_x = util.discretize(self.biasmap.shape[0], BIASMAP_XY_SUBDIV, start_state[0].getX())
        bi_y = util.discretize(self.biasmap.shape[1], BIASMAP_XY_SUBDIV, start_state[0].getY())
        bi_yaw = util.discretize(self.biasmap.shape[2], BIASMAP_YAW_SUBDIV, start_state[0].getYaw() / (2 * np.pi))
        result_data = self.biasmap[bi_x, bi_y, bi_yaw, :]
        result_valid = self.biasmap_valid[bi_x, bi_y, bi_yaw]
        result_exact = self.exact_flags[bi_x, bi_y, bi_yaw, :]
        if result_valid:
            if result_exact[0]:
                result_exact[0] = False
                for i in range(NUM_CONTROLS):
                    control[i] = np.clip(result_data[i], CONTROL_LOWER[i], CONTROL_UPPER[i])
            else:
                for i in range(NUM_CONTROLS):
                    cvalue = np.random.normal(result_data[i], BIASMAP_CONTROL_STDEV[i])
                    cvalue = np.clip(cvalue, CONTROL_LOWER[i], CONTROL_UPPER[i])
                    control[i] = cvalue
        else:
            for i in range(NUM_CONTROLS):
                control[i] = np.random.uniform(CONTROL_LOWER[i], CONTROL_UPPER[i])

class QCPlan1:
    def __init__(self, hardware_map, waypoints_fn, gridmap_fn, biasmap_fn):
        self.hardware_map = hardware_map
        
        self.waypoints = np.loadtxt(waypoints_fn, delimiter=',', dtype=np.float32)
        self.gridmap = np.load(gridmap_fn)
        self.biasmap_fn = biasmap_fn
        try:
            biasmap_data = np.load(biasmap_fn)
            self.biasmap = biasmap_data["biasmap"]
            self.biasmap_valid = biasmap_data["biasmap_valid"]
            print("====>", "Biasmap loaded from disk")
        except:
            self.biasmap = np.zeros((
                round(self.gridmap.shape[0] / GRIDMAP_XY_SUBDIV * BIASMAP_XY_SUBDIV) + 1,
                round(self.gridmap.shape[1] / GRIDMAP_XY_SUBDIV * BIASMAP_XY_SUBDIV) + 1,
                BIASMAP_YAW_SUBDIV + 1,
                NUM_CONTROLS,
            ), dtype=np.float32)
            self.biasmap_valid = np.zeros((
                round(self.gridmap.shape[0] / GRIDMAP_XY_SUBDIV * BIASMAP_XY_SUBDIV) + 1,
                round(self.gridmap.shape[1] / GRIDMAP_XY_SUBDIV * BIASMAP_XY_SUBDIV) + 1,
                BIASMAP_YAW_SUBDIV + 1,
            ), dtype=bool)
            print("====>", "Biasmap created fresh")

        self.se2space = ob.SE2StateSpace()
        self.se2space.setSubspaceWeight(0, 1) # R^2 subspace weight 1
        self.se2space.setSubspaceWeight(1, 0) # SO(2) subspace weight 0
        self.vectorspace = ob.RealVectorStateSpace(6)
        bounds = ob.RealVectorBounds(6)
        bounds.setLow(-99999) # don't care
        bounds.setHigh(99999)
        self.vectorspace.setBounds(bounds)

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

        self.planner = oc.SST(self.si)
        self.planner.setPruningRadius(0.01) # tenth of default
        self.planner.setSelectionRadius(0.02) # tenth of default
        self.ss.setPlanner(self.planner)

        #========

        self.last_physics_ticks_elapsed = 0
        self.last_control = [0, 0]
        self.control = None

        print("====>", "Waiting for hardware map")
        while not hardware_map.ready():
            rospy.sleep(0.001)
        
        self.state = ob.State(self.statespace)
        state = self.state()
        if hardware_map.scan.header.frame_id.startswith("ego"):
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
        
    def shutdown(self):
        print("====>", "Saving biasmap to disk")
        np.savez(self.biasmap_fn, biasmap=self.biasmap, biasmap_valid=self.biasmap_valid)

    def loop(self, timer):
        start = rospy.get_time()

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
        self.statespace.copyState(future_state(), self.state())
        if self.control is not None:
            self.state_propagate(self.state(), self.control, CHUNK_MULTIPLIER, future_state())
            self.last_control = self.control

        # Plan from future state
        self.ss.clear()
        self.ss.setStartState(future_state)
        start_point = np.array([future_state()[0].getX(), future_state()[0].getY()], dtype=np.float32)
        nearest_point, nearest_dist, t, i = util.nearest_point_on_trajectory(start_point, self.waypoints)
        goal_point, t, i = util.walk_along_trajectory(self.waypoints, t, i, CHUNK_DISTANCE)
        goal = ob.State(self.statespace)
        goal()[0].setX(goal_point[0])
        goal()[0].setY(goal_point[1])
        self.ss.setGoalState(goal, GOAL_THRESHOLD)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, min(goal_point[0], start_point[0]) - CHUNK_DISTANCE)
        bounds.setLow(1, min(goal_point[1], start_point[1]) - CHUNK_DISTANCE)
        bounds.setHigh(0, max(goal_point[0], start_point[0]) + CHUNK_DISTANCE)
        bounds.setHigh(1, max(goal_point[1], start_point[1]) + CHUNK_DISTANCE)
        self.se2space.setBounds(bounds)
        solved = self.ss.solve(CHUNK_DURATION - 0.010)
        if solved:
            solution = self.ss.getSolutionPath()
            controls = solution.getControls()
            states = solution.getStates()
            count = solution.getControlCount()
            for i in range(count):
                n_state = states[i]
                n_control = controls[i]
                bi_x = util.discretize(self.biasmap.shape[0], BIASMAP_XY_SUBDIV, n_state[0].getX())
                bi_y = util.discretize(self.biasmap.shape[1], BIASMAP_XY_SUBDIV, n_state[0].getY())
                bi_yaw = util.discretize(self.biasmap.shape[2], BIASMAP_YAW_SUBDIV, n_state[0].getYaw() / (2 * np.pi))
                for c in range(NUM_CONTROLS):
                    self.biasmap[bi_x, bi_y, bi_yaw, c] = n_control[c]
                self.biasmap_valid[bi_x, bi_y, bi_yaw] = True
            if count < MIN_PLAN_LENGTH:
                self.control = [controls[0][0], 0]
                print("====>", "Short plan, braking")
                #np.save(os.path.abspath(os.path.dirname(__file__)) + "/saved_map.npy", self.latched_map)
                #timer.shutdowm()
            else:
                self.control = [controls[0][0], controls[0][1]]
                print("====>", count, self.control)
        else:
            print("====>", "Not solved, zeroing controls")
            self.control = [0, 0]

        #print(rospy.get_time() - start)

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
            start[1][3], # start[1][4], start[1][5]
        ])

        steer = start[1][5]
        steer0 = start[1][4]
        vel = control[1]

        for i in range(int(duration)):
            # bound yaw angle
            if np_state[4] > 2*np.pi:
                np_state[4] = np_state[4] - 2*np.pi
            elif np_state[4] < 0:
                np_state[4] = np_state[4] + 2*np.pi

            # steering angle velocity input to steering velocity acceleration input
            accl, sv = util.pid(vel, steer, np_state[3], np_state[2], PARAMS['sv_max'], PARAMS['a_max'], PARAMS['v_max'], PARAMS['v_min'])
            
            # update physics, get RHS of diff'eq
            f = util.vehicle_dynamics_st(
                np_state,
                np.array([sv, accl]),
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
                PARAMS['v_max'])

            # update state
            np_state = np_state + f * PHYSICS_TIMESTEP
                
            steer = steer0
            steer0 = control[0]
            vel = control[1]
        
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
        return BiasmapControlSampler(control_space, self.biasmap, self.biasmap_valid)

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
        "%s/biasmap.npz" % filepath,
    )
    loop_timer = rospy.Timer(rospy.Duration(CHUNK_DURATION), qc.loop)
    rospy.spin()
