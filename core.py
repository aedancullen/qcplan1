import time
import math

import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

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
        start_point = np.array([start_state[0].getX(), start_state[0].getY()]), dtype=np.float32)
        nearest_point, nearest_dist, t, i = util.nearest_point_on_trajectory(start_point, waypoints)
        goal_point, t, i = util.walk_along_trajectory(waypoints, t, i, CHUNK_DISTANCE)
        self.goal_point = goal_point
        goal = ob.State(si)()
        goal[0].setX(goal_point[0])
        goal[0].setY(goal_point[1])
        self.setState(goal)

    def distanceGoal(start_state):
        start_point = np.array([start_state[0].getX(), start_state[0].getY()]), dtype=np.float32)
        nearest_point, nearest_dist, t, i = util.nearest_point_on_trajectory(start_point, waypoints)
        return np.linalg.norm(self.goal_point, nearest_point)

class TimestepOptimizationObjective(ob.OptimizationObjective):
    def __init__(self, si):
        super().__init__(si)
        self.si = si

    def motionCost(self, s1, s2):
        return 1 + self.si.distance(s1, s2)

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
        bi_yaw = round((start_state[0].getYaw() + math.pi) * BIASMAP_YAW_SUBDIV / (2 * math.pi))
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
    def __init__(self, agent_name, waypoints_fn, gridmap_fn, biasmap_fn):
        self.waypoints = np.loadtxt(waypoints_fn, delimiter=',', dtype=np.float32)
        self.gridmap = np.load(gridmap_fn)
        biasmap_data = np.load(biasmap_fn)
        self.biasmap = biasmap_data["biasmap"]
        self.biasmap_valid = biasmap_data["biasmap_valid"]

        self.se2space = ob.SE2StateSpace()
        self.vectorspace = ob.RealVectorStateSpace(5)

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
        self.si.setMinMaxControlDuration(1, 1)

        self.planner = oc.SST(self.si)
        self.ss.setPlanner(self.planner)

        self.ss.getProblemDefinition().setOptimizationObjective(TimestepOptimizationObjective(self.si))
        
        self.last_controls = None
        self.last_state = None
        self.last_scan = None

        self.provided_car = util.RaceCar(PARAMS, 12345) # seed garbage not used; nobody cares

    def loop(self):
        # On timer:
        
        # Issue prepared controls
        # Compute next position after chunk duration
        # Get latest scan, setup and plan for next chunk
        # Save prepared controls
        
    def lidar_callback(self):
        

    def state_validity_check(self, state):
        return self.si.satisfiesBounds(state)

    def state_propagate(self, start, control, duration, state):
        state[0].setX(start[0].getX() + control[0] * duration * cos(start[0].getYaw()))
        state[0].setY(start[0].getY() + control[0] * duration * sin(start[0].getYaw()))
        #state.setYaw(start.getYaw() + control[1] * duration)

    def csampler_alloc(self, control_space):
        return BiasmapControlSampler(control_space, self.biasmap, self.biasmap_valid)

def plan():
    # construct the state space we are planning in
    space = ob.SE2StateSpace()

    # set the bounds for the R^2 part of SE(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-1)
    bounds.setHigh(1)
    space.setBounds(bounds)

    # create a control space
    cspace = oc.RealVectorControlSpace(space, 2)

    # set the bounds for the control space
    cbounds = ob.RealVectorBounds(2)
    cbounds.setLow(-.3)
    cbounds.setHigh(.3)
    cspace.setBounds(cbounds)

    # define a simple setup class
    ss = oc.SimpleSetup(cspace)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn( \
        partial(state_validity_check, ss.getSpaceInformation())))
    ss.setStatePropagator(oc.StatePropagatorFn(state_propagate))

    # create a start state
    start = ob.State(space)
    start().setX(-0.5)
    start().setY(0.0)
    start().setYaw(0.0)

    # create a goal state
    goal = ob.State(space)
    goal().setX(0.0)
    goal().setY(0.5)
    goal().setYaw(0.0)

    # set the start and goal states
    ss.setStartAndGoalStates(start, goal, 0.1)

    # (optionally) set planner
    si = ss.getSpaceInformation()
    planner = oc.SST(si)
    #planner = oc.EST(si)
    #planner = oc.KPIECE1(si) # this is the default
    # SyclopEST and SyclopRRT require a decomposition to guide the search
    #decomp = MyDecomposition(32, bounds)
    #planner = oc.SyclopEST(si, decomp)
    #planner = oc.SyclopRRT(si, decomp)
    ss.setPlanner(planner)
    # (optionally) set propagation step size
    si.setPropagationStepSize(1)
    si.setMinMaxControlDuration(1, 1)

    ss.getControlSpace().setControlSamplerAllocator(oc.ControlSamplerAllocator(csampler_alloc))
    ss.getProblemDefinition().setOptimizationObjective(TimestepOptimizationObjective(si))

    # attempt to solve the problem
    solved = ss.solve(0.100)

    if solved:
        # print the path to screen
        print("Found solution:\n%s" % ss.getSolutionPath().printAsMatrix())

if __name__ == "__main__":
    qc = QCPlan1()
    loop_timer = rospy.Timer(rospy.Duration(CHUNK_DURATION), qc.loop)
    scan_sub = rospy.Subscriber("/%s/scan" % self.agent_name, LaserScan, qc.lidar_callback, queue_size=1)
    rospy.spin()
