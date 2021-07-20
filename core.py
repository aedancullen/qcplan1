import time
import math
from functools import partial

import numpy as np
from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

import util

#ou.setLogLevel(ou.LOG_WARN)

PARAMS = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

NUM_CONTROLS = 2
CONTROL_LOWER = [0.0, 0.0]
CONTROL_UPPER = [1.0, 1.0]

BIASMAP_XY_SUBDIV = 10
BIASMAP_YAW_SUBDIV = 10
BIASMAP_CONTROL_STDEV = [1.0, 1.0]

class CourseProgressGoal(ob.GoalState):
    def __init__(self, si, waypoints, start_state, progress_dist):
        super().__init__(si)
        self.si = si
        self.waypoints = waypoints
        start_point = np.array([start_state.getX(), start_state.getY()]), dtype=np.float32)
        nearest_point, nearest_dist, t, i = util.nearest_point_on_trajectory(start_point, waypoints)
        goal_point, t, i = util.walk_along_trajectory(waypoints, t, i, progress_dist)
        self.goal_point = goal_point
        goal = ob.State(si)
        goal().setX(goal_point[0])
        goal().setY(goal_point[1])
        self.setState(goal)
        
    def distanceGoal(start_state):
        start_point = np.array([start_state.getX(), start_state.getY()]), dtype=np.float32)
        nearest_point, nearest_dist, t, i = util.nearest_point_on_trajectory(start_point, waypoints)
        return np.linalg.norm(self.goal_point, nearest_point)

class TimestepOptimizationObjective(ob.OptimizationObjective):
    def __init__(self, si):
        super().__init__(si)
        self.si = si
        
    def motionCost(self, s1, s2):
        return 1 + self.si.distance(s1, s2)
    
class BiasmapControlSampler(oc.ControlSampler):
    def __init__(self, control_space, biasmap, biasmap_valid):
        super().__init__(control_space)
        self.control_space = control_space
        self.biasmap = biasmap
        self.biasmap_valid = biasmap_valid
        
    def sample(self, control, start_state):
        bi_x = round(start_state.getX() * BIASMAP_XY_SUBDIV)
        bi_y = round(start_state.getY() * BIASMAP_XY_SUBDIV)
        bi_yaw = round((start_state.getYaw() + math.pi) * BIASMAP_YAW_SUBDIV / (2 * math.pi))
        result_data = self.biasmap[bi_x, bi_y, bi_yaw, :, :]
        result_valid = self.biasmap_valid[bi_x, bi_y, bi_yaw]
        if result_valid:
            for i in range(NUM_CONTROLS):
                cvalue = np.random.normal(result_data[i, 0], result_data[i, 1])
                cvalue = np.clip(cvalue, CONTROL_LOWER[i], CONTROL_UPPER[i])
                control[i] = cvalue
        else:
            for i in range(NUM_CONTROLS):
                control[i] = np.random.uniform(CONTROL_LOWER[i], CONTROL_UPPER[i])

def state_validity_check(spaceInformation, state):
    return spaceInformation.satisfiesBounds(state)

def state_propagate(start, control, duration, state):
    state.setX(start.getX() + control[0] * duration * cos(start.getYaw()))
    state.setY(start.getY() + control[0] * duration * sin(start.getYaw()))
    #state.setYaw(start.getYaw() + control[1] * duration)
    
def csampler_alloc(controlSpace):
    return oc.RealVectorControlUniformSampler(controlSpace)

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
    plan()
    s = time.time()
    plan()
    print(time.time() - s)
    #while True:
    #    time.sleep(0.100)
    #    print("yo")
