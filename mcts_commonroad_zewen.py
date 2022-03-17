__author__ = "Zewen Chen"
__copyright__ = "TUM Chair of Robotics, Artificial Intelligence and Real-time Systems"
__version__ = "2022.3"
__maintainer__ = "Chi Zhang"
__email__ = "ge96vij@mytum.de"


from copy import deepcopy
from scipy.spatial import distance
from shapely.geometry import Polygon
import numpy as np
import math
import time
import random

from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import State, Trajectory

class EgoState:
    def __init__(self, velocity=0, position=np.array([0.0, 0.0]), orientation=0, time_step=0, shape=Rectangle(length=4.5, width=2), goal=None):
        self.velocity = velocity
        self.position = position
        self.orientation = orientation # The orientation is in radians, converted into an angle as "redians * 180 / Pi"
        self.time_step = time_step
        self.shape = shape
        self.goal = goal

        
class VehicleState:
    def __init__(self):
        self.velocity = 0
        self.position = np.array([0.0, 0.0])
        self.orientation = 0
        self.ID = 0
        
class Status:
    def __init__(self):
        self.is_reached = False
        self.is_collided = False


"""
define the state space and member functions
"""
class StateSpace:
    def __init__(self):
        self.ego = EgoState()
        self.vehicles = VehicleState()
        self.status = Status()
        
    def getPossibleActions(self):
        return [-5, -2.5, -1.5, 0.0, 1.5, 2.5]
        
    def takeAction(self, action, dt = 0.1):
        state = deepcopy(self)
        
        # Update next ego state
        state.ego = self.calculateNextEgoState(state, action, dt)
        
        # Update next vehicles state
        state.vehicle = self.calculateNextVehicleState(state, dt)
        
        # Update status between ego and vehicles
        state.status = self.calculateNextStatus(state)
        
        return state
    
    def calculateNextEgoState(self, state, action, dt):
        
        if state.ego.velocity < 0:
            state.ego.velocity = 0
            
        state.ego.position[0] += (state.ego.velocity * dt + 0.5 * action * dt ** 2) *  math.cos(math.radians(state.ego.orientation) / (math.pi / 2) * 90)
        state.ego.position[1] += (state.ego.velocity * dt + 0.5 * action * dt ** 2) *  math.sin(math.radians(state.ego.orientation) / (math.pi / 2) * 90)
        waypoint_ego, index = self.getCurrentEgoWaypoint(state)
        state.ego.position = waypoint_ego.position
        state.ego.orientation = waypoint_ego.orientation
        state.ego.velocity += action * dt
        
        return state.ego
    
    def calculateNextVehicleState(self, state, dt):
        # Vehicles is a list of instances
        # Predict the state of vehicles
                
        return state.vehicles
    
    
    def calculateNextStatus(self, state):
        if state.ego.goal.is_reached(state.ego):
            state.status.is_reached = True
            
        """
        1. Check for collision of two vehicles with the same radius. 
        The middle points of circles are their positions. Touching circles count also as collision
        
        
        radius = math.sqrt((state.ego.shape.length / 2) ** 2 + (state.ego.shape.width / 2) ** 2)
        for vehicle in state.vehicles:
            distance = math.sqrt((state.ego.position[0] - vehicle.position[0]) ** 2 + (state.ego.position[1] - vehicle.position[1]) ** 2)
            if distance <= 2 * radius:
                state.status.is_collided = True
                break
        return state.status
    
        """
        
        
        """
        2. Calculate the shapely polygon, check if polygon areas overlap
        """
        vehicle_points = np.array([[state.ego.shape.length / 2, state.ego.shape.width / 2],
                                  [state.ego.shape.length / 2, -state.ego.shape.width / 2],
                                  [-state.ego.shape.length / 2, -state.ego.shape.width / 2],
                                  [-state.ego.shape.length / 2, state.ego.shape.width / 2]])
        
        polygon_ego = self.getPolygon(vehicle_points, state.ego.position[0], state.ego.position[1], state.ego.orientation)
        polygon_vehicles = []
        for vehicle in state.vehicles:
            polygon_vehicle = self.getPolygon(vehicle_points, vehicle.position[0], vehicle.position[1], vehicle.orientation)
            polygon_vehicles.append(polygon_vehicle)
            
        for polygon_vehicle in polygon_vehicles:
            if polygon_ego.intersects(polygon_vehicle):
                state.status.is_collided = True
                break
                
        return state.status
    
    def getPolygon(self, vehicle_points, x_tra, y_tra, orientation):
        xy_list_rot_tra = []
        for i, vehicle_points in enumerate(vehicle_points):
            x_rot = vehicle_points[0] * math.cos(math.radians(orientation / (math.pi / 2) * 90)) - vehicle_points[1] * math.sin(math.radians(orientation / (math.pi / 2) * 90))
            y_rot = vehicle_points[0] * math.sin(math.radians(orientation / (math.pi / 2) * 90)) + vehicle_points[1] * math.cos(math.radians(orientation / (math.pi / 2) * 90))
            
            xy_list_rot_tra.append((x_rot + x_tra, y_rot + y_tra))
            
        return Polygon(xy_list_rot_tra)
    
    def getCurrentEgoWaypoint(self, state):
        waypoints = CommonroadEnv(scenario, planning_problem_set).generate_waypoint_ego()
        distance = []
        for i in range(len(waypoints)):
            d = np.linalg.norm(state.ego.position - waypoints[i].position) 
            distance.append(d)

        return waypoints[distance.index(min(distance))], distance.index(min(distance))
    
    def getCurrentVehicleWaypoint(self, vehicle):
        waypoints = CommonroadEnv(scenario, planning_problem_set).generate_waypoint_vehicle(vehicle)
        return 0
    
    def isTerminal(self):
        return self.status.is_reached or self.status.is_collided
    
    def getReward(self, action=None):
        ego_speed = self.ego.velocity
        # if collision, reward -2000-v^2; if success, reward 1000
        if self.isTerminal():
            return -2000 - ego_speed ** 2 if self.status.is_collided else 1000
        # speed reward: 0 reward when 8.33 ~ 11.11 m/s(30 ~ 40 km/h); out of the range negative reward
        speed_reward = 1.0 * (ego_speed - 8.33) if ego_speed < 8.33 else (
            0 if ego_speed < 11.11 else 4 * (11.11 - ego_speed))
        # action reward
        action_reward = -0.1 if action != 0 else 0

        return speed_reward + action_reward


def randomPolicy(state):
    accum_reward = 0
    for i in range(6):
        if not state.isTerminal():
            action = random.choice(state.getPossibleActions())
            state = state.takeAction(action)
            accum_reward += state.getReward(action)
        else:
            return state.getReward()+accum_reward
    return accum_reward

class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):
        s = []
        s.append("totalReward: %s" % (self.totalReward))
        s.append("numVisits: %d" % (self.numVisits))
        s.append("isTerminal: %s" % (self.isTerminal))
        s.append("possibleActions: %s" % (self.children.keys()))
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))


class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState, needDetails=False):
        self.root = treeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        action = (action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def executeRound(self):
        """
        execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)


import os
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from IPython.display import clear_output

# import classes and functions for reading xml file and visualizing commonroad objects
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad.common.util import Interval

# generate path of the file to be read
path_file = "/Users/chenzewen/Downloads/commonroad-search-master/scenarios/exercise/USA_Peach-2_1_T-1.xml"

# read in the scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(path_file).open()
    

class Waypoint():
    def __init__(self, position=np.array([0.0, 0.0]), orientation=0, length=0):
        self.position = position
        self.orientation = orientation
        self.length = length       


class CommonroadEnv():
    def __init__(self, scenario, planning_problem_set):
        self.scenario = scenario
        self.planning_problem_set = planning_problem_set
        self.state = StateSpace()
        self.time_step = 0
        self.dt = 0.1
        self.reward = 0
        self.done = False
        self.action = 0
        self.ego_initial_state = self.generate_ego_initial_state()
        self.vehicles_initial_state = self.generate_vehicles_initial_state()
        self.trajectory_state_list = [State(position = self.ego_initial_state.position, orientation = self.ego_initial_state.orientation, time_step = self.time_step)]
        
        
    def generate_ego_initial_state(self):
        for key in self.planning_problem_set.planning_problem_dict.keys():
            # self.goal = planning_problem_set.planning_problem_dict[key].goal
            ego_initial_state = EgoState(velocity = self.planning_problem_set.planning_problem_dict[key].initial_state.velocity,
                                         position = self.planning_problem_set.planning_problem_dict[key].initial_state.position, 
                                         orientation = self.planning_problem_set.planning_problem_dict[key].initial_state.orientation, 
                                         time_step = self.planning_problem_set.planning_problem_dict[key].initial_state.time_step,
                                         goal = self.planning_problem_set.planning_problem_dict[key].goal)
            
        return ego_initial_state
    
    def generate_vehicles_initial_state(self):
        vehicles_initial_state = []
        for i in range(len(self.scenario.dynamic_obstacles)):
            vehicle = VehicleState()
            vehicle.position = self.scenario.dynamic_obstacles[i].initial_state.position
            vehicle.orientation = self.scenario.dynamic_obstacles[i].initial_state.orientation
            vehicle.velocity = self.scenario.dynamic_obstacles[i].initial_state.velocity
            vehicle.ID = self.scenario.dynamic_obstacles[i].obstacle_id
            vehicles_initial_state.append(vehicle)
        
        return vehicles_initial_state
        
    def generate_waypoint_ego(self) -> list:
        # generate waypoints from ego
        route_planner_ego = RoutePlanner(self.scenario, list(self.planning_problem_set.planning_problem_dict.values())[0], backend=RoutePlanner.Backend.NETWORKX_REVERSED)
        candidate_holder_ego = route_planner_ego.plan_routes()
        route_ego = candidate_holder_ego.retrieve_best_route_by_orientation() # choose an optimal route
        waypoints_ego = []
        
        for i in range(len(route_ego.reference_path)):
            waypoint = Waypoint()
            waypoint.position = route_ego.reference_path[i]
            waypoint.orientation = route_ego.path_orientation[i]
            waypoint.length = route_ego.path_length
            waypoints_ego.append(waypoint)
        
        return waypoints_ego
    
    def generate_waypoint_vehicle(self, vehicle) -> dict:
        # generate waypoints from vehicles
        state_vehicle = State(position=vehicle.initial_state.position, 
                              velocity=vehicle.initial_state.velocity, 
                              orientation=vehicle.initial_state.orientation,
                              yaw_rate=0,
                              slip_angle=0,
                              time_step=0)

        planning_problem_vehicle = PlanningProblem(planning_problem_id=list(self.planning_problem_set.planning_problem_dict.values())[0].planning_problem_id, 
                                                   initial_state=state_vehicle, 
                                                   goal_region=GoalRegion(state_list=[State(time_step=Interval(start=99,end=100))]))
            
        route_planner_vehicle = RoutePlanner(self.scenario, planning_problem_vehicle, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
        candidate_holder_vehicle = route_planner_vehicle.plan_routes()
        waypoints_vehicle = {}
        waypoints_from_one_route = []
            
        for i, route_vehicle in enumerate(candidate_holder_vehicle.list_route_candidates):
            for j in range(len(route_vehicle.reference_path)):
                waypoint = Waypoint()
                waypoint.position = route_vehicle.reference_path[j]
                waypoint.orientation = route_vehicle.path_orientation[j]
                waypoint.length = route_vehicle.path_length
                waypoints_from_one_route.append(waypoint)
                
            waypoints_vehicle[i] = waypoints_from_one_route
            
        return waypoints_vehicle
    
    def start(self):
        self.state.ego = self.ego_initial_state
        self.state.vehicles = self.vehicles_initial_state
        
        for i in range(0, 20):
            plt.figure(figsize=(10, 10))
            renderer = MPRenderer()

            # uncomment the following line to visualize with animation
            clear_output(wait=True)

            # plot the scenario for each time step
            scenario.draw(renderer, draw_params={'time_begin': i})

            # plot the planning problem set
            planning_problem_set.draw(renderer)

            renderer.render()
            plt.show()
        
        return self.state
    
    def step(self, action):
        self.action = action
        
        """
        Bring the speed or action of the ego into the sumo simulator
        
        """
        
        self.state.ego.position[0] += (self.state.ego.velocity * self.dt + 0.5 * action * self.dt ** 2) * math.cos(math.radians(self.state.ego.orientation) / (math.pi / 2) * 90)
        self.state.ego.position[1] += (self.state.ego.velocity * self.dt + 0.5 * action * self.dt ** 2) * math.sin(math.radians(self.state.ego.orientation) / (math.pi / 2) * 90)
        
        self.state.ego.velocity = self.state.ego.velocity + action * self.dt
        if self.state.ego.velocity < 0:
            self.state.ego.velocity = 0
        self.time_step = self.time_step + 1
        
        for vehicle in self.state.vehicles:
            vehicle.position = self.get_obstacle_state_at_time()['dynamic'][vehicle.ID].position
            vehicle.orientation = self.get_obstacle_state_at_time()['dynamic'][vehicle.ID].orientation
            vehicle.velocity = self.get_obstacle_state_at_time()['dynamic'][vehicle.ID].velocity
            
        self.done = self.is_done()
        self.reward = self.get_reward()
            
        return self.state, self.reward, self.done
    
    def reset(self):
        self.time_step = 0
        self.reward = 0
        self.done = False
        self.state.ego = deepcopy(self.ego_initial_state)
        self.state.vehicles = deepcopy(self.vehicles_initial_state)
        self.trajectory_state_list = [State(position = self.ego_initial_state.position, orientation = self.ego_initial_state.orientation, time_step = self.time_step)]
        
        return self.state
        
    def get_reward(self):
        # if collision, reward -2000-v^2; if success, reward 1000
        if self.done:
            self.reward = -2000 - self.state.ego.velocity ** 2 if self.is_collided() else 1000
            return self.reward
        # speed reward: 0 reward when 8.33 ~ 11.11 m/s(30 ~ 40 km/h); out of the range negative reward
        
        speed_reward = 1.0 * (self.state.ego.velocity - 8.33) if self.state.ego.velocity < 8.33 else (
            0 if self.state.ego.velocity < 11.11 else 4 * (11.11 - self.state.ego.velocity))
        # action reward
        action_reward = -0.1 if self.action != 0 else 0

        self.reward = speed_reward + action_reward
        return self.reward
    
    def is_done(self):
        return self.is_reached() or self.is_collided()
    
    def is_reached(self):
        return self.ego_initial_state.goal.is_reached(self.state.ego)
    
    def is_collided(self):
        cc = create_collision_checker(scenario)

        new_state = State(position = self.state.ego.position, orientation = self.state.ego.orientation, time_step = self.time_step)
        self.trajectory_state_list.append(new_state)    
        ego_trajectory = Trajectory(0, self.trajectory_state_list)
        
        # create a TrajectoryPrediction object consisting of the trajectory and the shape of the ego vehicle
        traj_pred = TrajectoryPrediction(trajectory=ego_trajectory, shape=self.state.ego.shape)
        co = create_collision_object(traj_pred)
        
        """
        rnd = MPRenderer(figsize=(35, 25))
        scenario.lanelet_network.draw(rnd)
        cc.draw(rnd, draw_params={'facecolor': 'blue'})
        co.draw(rnd, draw_params={'facecolor': 'green'})
        rnd.render()
        plt.show()
        """
        return cc.collide(co)
        
    def get_obstacle_state_at_time(self):
        # Returns a dictionary containing static and dynamic obstacles
        obstacle_states = {'dynamic':{}, 'static':{}}

        for obstacle in scenario.dynamic_obstacles:
            if obstacle.state_at_time(self.time_step) is not None:
                obstacle_states['dynamic'][obstacle.obstacle_id] = obstacle.state_at_time(self.time_step)
            else:
                obstacle_states['dynamic'][obstacle.obstacle_id] =  State(position=np.array([-999,-999]), 
                                                                          orientation=0, 
                                                                          velocity=0, 
                                                                          acceleration=0,
                                                                          time_step=self.time_step)
        for obstacle in scenario.static_obstacles:
            obstacle_states['static'][obstacle.obstacle_id] = obstacle.initial_state

        return obstacle_states


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

if __name__ == "__main__":

    env = CommonroadEnv(scenario, planning_problem_set)
    env.start()
    episodes = [i for i in range(1)]
    scores = []
    average_scores = []
    scores_window = deque(maxlen=10)
    actions = [-5, -2.5, -1.5, 0.0, 1.5, 2.5]

    # run episode
    for episode in episodes:
        score = 0
        state = env.reset()
        searcher = mcts(iterationLimit=10)

        while True:
            mcts_state = state
            action = searcher.search(initialState=mcts_state)
            state, reward, done = env.step(action)
            print(reward, env.is_collided(), env.state.ego.velocity)
            score += reward
            
            if done:
                scores.append(score)
                average_scores.append(sum(scores) / (episode + 1))
                print("episode: {}, score: {:.2f}".format(episode, score))
                break
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(scores)), scores, label = 'score')
    ax.plot(np.arange(len(scores)), average_scores, label = 'average score')
    ax.legend()
    plt.ylabel('score')
    plt.xlabel('Episode')
    plt.show()


