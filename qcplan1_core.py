
import time
from math import pi, sin, cos
from functools import partial

import numpy as np
from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

#ou.setLogLevel(ou.LOG_WARN)

NUM_CONTROLS = 2
CONTROL_LOWER = [0.0, 0.0]
CONTROL_UPPER = [1.0, 1.0]

BIASMAP_XY_SUBDIV = 1000/50
BIASMAP_YAW_SUBDIV = 10
BIASMAP_CONTROL_STDEV = [1.0, 1.0]

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
        self.exact_flags = np.ones_like(biasmap, dtype=bool)
        
    def sample(self, control, state):
        bi_x, bi_y, bi_yaw = se2_to_biasmap_indices(state)
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
                
def se2_to_biasmap_indices(state):
    bi_x = round(state.getX() * BIASMAP_XY_SUBDIV)
    bi_y = round(state.getY() * BIASMAP_XY_SUBDIV)
    bi_yaw = round((state.getYaw() + pi) * BIASMAP_YAW_SUBDIV / (2 * pi))
    return bi_x, bi_y, bi_yaw

def state_validity_check(spaceInformation, state):
    return spaceInformation.satisfiesBounds(state)

def state_propagate(start, control, duration, state):
    state.setX(start.getX() + control[0] * duration * cos(start.getYaw()))
    state.setY(start.getY() + control[0] * duration * sin(start.getYaw()))
    state.setYaw(start.getYaw() + control[1] * duration)
    
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
    goal().setYaw(-4)

    # set the start and goal states
    ss.setStartAndGoalStates(start, goal, 0.05)

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
    
    print("Begin solve")
    # attempt to solve the problem
    solved = ss.solve(20.100)
    
    ss.clear()
    
    solved = ss.solve(20.100)

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
