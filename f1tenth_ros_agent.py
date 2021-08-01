#!/usr/bin/python3
import os

import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from f1tenth_gym_ros.msg import RaceInfo

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

BIASMAP_XY_SUBDIV = 10
BIASMAP_YAW_SUBDIV = 20

PHYSICS_TIMESTEP = 0.01 # Actual value used in calculation
SIM_INTERVAL = 0.02 # Real time interval of simulator's internal physics callbacks

CHUNK_MULTIPLIER = 10

CHUNK_DURATION = SIM_INTERVAL * CHUNK_MULTIPLIER
CHUNK_DISTANCE = 10

class CourseProgressGoal(ob.GoalState):
    def __init__(self, si, waypoints, start_state):
        super().__init__(si)
        self.si = si
        self.waypoints = waypoints
        start_point = np.array([start_state[0].getX(), start_state[0].getY()], dtype=np.float32)
        nearest_point, nearest_dist, t, i = util.nearest_point_on_trajectory(start_point, waypoints)
        goal_point, t, i = util.walk_along_trajectory(waypoints, t, i, CHUNK_DISTANCE)
        self.goal_point = goal_point
        goal = ob.State(si.getStateSpace())
        goal()[0].setX(goal_point[0])
        goal()[0].setY(goal_point[1])
        self.setState(goal)

    def distanceGoal(start_state):
        start_point = np.array([start_state[0].getX(), start_state[0].getY()], dtype=np.float32)
        nearest_point, nearest_dist, t, i = util.nearest_point_on_trajectory(start_point, waypoints)
        return np.linalg.norm(self.goal_point, nearest_point)

class TimestepOptimizationObjective(ob.OptimizationObjective):
    def __init__(self, si):
        super().__init__(si)
        self.si = si

    def motionCost(self, s1, s2):
        return 1# + self.si.distance(s1, s2)

class BiasmapControlSampler(oc.ControlSampler):
    def __init__(self, controlspace, biasmap, biasmap_valid):
        super().__init__(controlspace)
        self.controlspace = controlspace
        self.biasmap = biasmap
        self.biasmap_valid = biasmap_valid
        self.exact_flags = np.ones_like(biasmap, dtype=bool)

    def sample(self, control, start_state):
        bi_x = round(start_state[0].getX() * BIASMAP_XY_SUBDIV)
        bi_y = round(start_state[0].getY() * BIASMAP_XY_SUBDIV)
        bi_yaw = round((start_state[0].getYaw() + np.pi) * BIASMAP_YAW_SUBDIV / (2 * np.pi))
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
        #self.gridmap = np.load(gridmap_fn)
        #biasmap_data = np.load(biasmap_fn)
        #self.biasmap = biasmap_data["biasmap"]
        #self.biasmap_valid = biasmap_data["biasmap_valid"]

        self.se2space = ob.SE2StateSpace()
        self.vectorspace = ob.RealVectorStateSpace(6)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, -5)
        bounds.setLow(1, -5)
        bounds.setHigh(0, 5)
        bounds.setHigh(1, 5)
        #self.vectorspace.setBounds(bounds)

        self.statespace = ob.CompoundStateSpace()
        self.statespace.addSubspace(self.se2space, 1)
        self.statespace.addSubspace(self.vectorspace, 0)

        self.controlspace = oc.RealVectorControlSpace(self.statespace, 2)
        self.controlspace.setControlSamplerAllocator(oc.ControlSamplerAllocator(self.csampler_alloc))

        self.ss = oc.SimpleSetup(self.controlspace)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.state_validity_check))
        self.ss.setStatePropagator(oc.StatePropagatorFn(self.state_propagate))

        self.si = self.ss.getSpaceInformation()
        self.si.setPropagationStepSize(1)
        self.si.setMinMaxControlDuration(CHUNK_MULTIPLIER, CHUNK_MULTIPLIER)

        self.planner = oc.SST(self.si)
        self.ss.setPlanner(self.planner)

        self.ss.getProblemDefinition().setOptimizationObjective(TimestepOptimizationObjective(self.si))
        
        #========
        
        self.last_physics_ticks_elapsed = 0
        self.last_control = [0, 0]
        self.control = None
        
        while not hardware_map.ready():
            rospy.sleep(0.001)
        
        self.state = ob.State(self.statespace)
        state = self.state()
        if hardware_map.scan.header.frame_id.startswith("ego"):
            i = 0
        else:
            i = 1
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

        physics_ticks_elapsed = round(self.hardware_map.race_info.ego_elapsed_time / PHYSICS_TIMESTEP)
        physics_ticks_new = physics_ticks_elapsed - self.last_physics_ticks_elapsed
        self.last_physics_ticks_elapsed = physics_ticks_elapsed
        
        self.state_propagate(self.state(), self.last_control, physics_ticks_new, self.state())
        
        future_state = self.state
        if self.control is not None:
            future_state = ob.State(self.statespace)
            self.state_propagate(self.state(), self.control, CHUNK_MULTIPLIER, future_state())
            self.last_control = self.control

        # Plan from future_state and save plan in self.control
        self.ss.clear()
        self.ss.setStartState(future_state)
        goal = CourseProgressGoal(self.si, self.waypoints, future_state())
        #start_point = np.array([future_state[0].getX(), future_state[0].getY()], dtype=np.float32)
        #nearest_point, nearest_dist, t, i = util.nearest_point_on_trajectory(start_point, self.waypoints)
        #goal_point, t, i = util.walk_along_trajectory(self.waypoints, t, i, CHUNK_DISTANCE)
        #goal = ob.State(self.statespace)
        #goal()[0].setX(goal_point[0])
        #goal()[0].setY(goal_point[1])
        self.ss.setGoal(goal)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, -5)
        bounds.setLow(1, -5)
        bounds.setHigh(0, 5)
        bounds.setHigh(1, 5)
        self.se2space.setBounds(bounds)
        solved = self.ss.solve(CHUNK_DURATION - 0.010)
        
        print(future_state()[0].getX())
        self.control = [0, 1]

    def state_validity_check(self, state):
        return self.si.satisfiesBounds(state)

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
        
        steer = start[1][5]
        steer0 = start[1][4]
        vel = control[1]
        
        for i in range(int(duration)):
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

            # bound yaw angle
            if np_state[4] > 2*np.pi:
                np_state[4] = np_state[4] - 2*np.pi
            elif np_state[4] < 0:
                np_state[4] = np_state[4] + 2*np.pi
                
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

    def csampler_alloc(self, control_space):
        return BiasmapControlSampler(control_space, self.biasmap, self.biasmap_valid)

class HardwareMap:
    def __init__(self):
        self.scan = None
        self.race_info = None

        self.scan_sub = rospy.Subscriber("/%s/scan" % agent_name, LaserScan, self.scan_callback, queue_size=1)
        self.race_info_sub = rospy.Subscriber("/race_info", RaceInfo, self.race_info_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/%s/drive" % agent_name, AckermannDriveStamped, queue_size=1)

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
        return self.scan is not None and self.race_info is not None

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
