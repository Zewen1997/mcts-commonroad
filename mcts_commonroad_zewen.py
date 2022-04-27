__author__ = "Zewen Chen"
__copyright__ = "TUM Chair of Robotics, Artificial Intelligence and Real-time Systems"
__version__ = "2022.4"
__maintainer__ = "Chi Zhang"
__email__ = "ge96vij@mytum.de"


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
path_file = os.path.abspath('scenarios/exercise/DEU_Flensburg-11_1_T-1.xml')

# read in the scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(path_file).open()
planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]



from copy import deepcopy
from datetime import datetime
from scipy.spatial import distance
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.ops import cascaded_union
import numpy as np
import math
import time
import random

from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.scenario.obstacle import Obstacle, StaticObstacle, ObstacleType
from SMP.motion_planner.utility import visualize_solution
from commonroad.common.solution import Solution, PlanningProblemSolution, VehicleModel, VehicleType, CostFunction
from commonroad.common.solution import CommonRoadSolutionWriter



class Waypoint():
    def __init__(self, position=np.array([0.0, 0.0]), orientation=0, path_length=0, curvature=0):
        self.position = position
        self.orientation = orientation
        self.path_length = path_length
        self.curvature = curvature



def generate_waypoints_ego() -> list:
    # generate waypoints from ego
    route_planner_ego = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
    candidate_holder_ego = route_planner_ego.plan_routes()
    route_ego = candidate_holder_ego.retrieve_best_route_by_orientation() # choose an optimal route
    waypoints_ego = []

    for i in range(len(route_ego.reference_path)):
        waypoint = Waypoint()
        waypoint.position = route_ego.reference_path[i]
        waypoint.orientation = route_ego.path_orientation[i]
        waypoint.path_length = route_ego.path_length[i]
        waypoint.curvature = route_ego.path_curvature[i]
        waypoints_ego.append(waypoint)

    return waypoints_ego
    
def generate_waypoints_vehicles() -> dict:
    waypoints_vehicles = {}
    for obstacle in scenario.dynamic_obstacles:
        state_vehicle = State(position=obstacle.initial_state.position, 
                                  velocity=obstacle.initial_state.velocity, 
                                  orientation=obstacle.initial_state.orientation,
                                  yaw_rate=0,
                                  slip_angle=0,
                                  time_step=0)
        planning_problem_vehicle = PlanningProblem(planning_problem_id=planning_problem.planning_problem_id, 
                                                       initial_state=state_vehicle, 
                                                       goal_region=GoalRegion(state_list=[State(time_step=Interval(start=99,end=100))]))
        route_planner_vehicle = RoutePlanner(scenario, planning_problem_vehicle, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
        candidate_holder_vehicle = route_planner_vehicle.plan_routes()
        waypoints_from_one_route = {}

        for i, route_vehicle in enumerate(candidate_holder_vehicle.list_route_candidates):
            waypoints = []
            for j in range(len(route_vehicle.reference_path)):
                waypoint = Waypoint()
                waypoint.position = route_vehicle.reference_path[j]
                waypoint.orientation = route_vehicle.path_orientation[j]
                waypoint.path_length = route_vehicle.path_length[j]
                waypoints.append(waypoint)
            waypoints_from_one_route[i] = waypoints

            waypoints_vehicles[obstacle.obstacle_id] = waypoints_from_one_route

    return waypoints_vehicles

_waypoints_ego = generate_waypoints_ego()
_waypoints_vehicles = generate_waypoints_vehicles()

def get_ego_initial_path_length(waypoints_ego):
    _distance = []
    key = list(planning_problem_set.planning_problem_dict.keys())[0]
    for i in range(len(waypoints_ego)):
        d = distance.euclidean(planning_problem_set.planning_problem_dict[key].initial_state.position, waypoints_ego[i].position)
        _distance.append(d)

    return _waypoints_ego[_distance.index(min(_distance))].path_length, _distance.index(min(_distance))

initial_ego_path_length, i = get_ego_initial_path_length(_waypoints_ego)
list_s = []
for s in _waypoints_ego:
    list_s.append(s.path_length)
    
list_s = list_s[i:]
_waypoints_ego = _waypoints_ego[i:]



def get_vehicle_initial_path(waypoints_vehicles):
    _distance = {}
    for obstacle in scenario.dynamic_obstacles:
        a = {}
        for k in range(len(waypoints_vehicles[obstacle.obstacle_id])):
            _distance1 = []
            for j in range(len(waypoints_vehicles[obstacle.obstacle_id][k])):
                d = distance.euclidean(obstacle.initial_state.position, waypoints_vehicles[obstacle.obstacle_id][k][j].position)
                _distance1.append(d)
            a[k] = [waypoints_vehicles[obstacle.obstacle_id][k][_distance1.index(min(_distance1))].path_length, _distance1.index(min(_distance1))]
        _distance[obstacle.obstacle_id] = a

    return _distance



list_s_v = {}
for obstacle in scenario.dynamic_obstacles:
    length = {}
    for x in range(len(_waypoints_vehicles[obstacle.obstacle_id])):
        _length = []
        for y in _waypoints_vehicles[obstacle.obstacle_id][x]:
            _length.append(y.path_length)
        length[x] = _length
    list_s_v[obstacle.obstacle_id] = length



dic = get_vehicle_initial_path(_waypoints_vehicles)

# add virtual obstacle for reaching the target
center_line = planning_problem.goal.state_list[0].position.shapes[0]._max
_distance = []
for i in range((len(_waypoints_ego))):
    d = distance.euclidean(center_line, _waypoints_ego[i].position)
    _distance.append(d)

virtual_obstacle = StaticObstacle(0, ObstacleType.CAR, Rectangle(3, 1.5, np.array([0, 0])), State(position=_waypoints_ego[_distance.index(min(_distance))].position, orientation=_waypoints_ego[_distance.index(min(_distance))].orientation, time_step=0, velocity=0))
scenario.add_objects(virtual_obstacle, list(planning_problem.goal.lanelets_of_goal_position.values())[0][0])



class EgoState:
    def __init__(self, velocity=0, position=np.array([0.0, 0.0]), orientation=0, time_step=0, shape=Rectangle(length=4.3, width=1.8), goal=None, trajectory=None, path_length=initial_ego_path_length, steering_angle=0):
        self.velocity = velocity
        self.position = position
        self.orientation = orientation # The orientation is in radians, converted into an angle as "redians * 180 / Pi"
        self.time_step = time_step
        self.shape = shape
        self.goal = goal
        self.trajectory = trajectory
        self.path_length = path_length
        self.steering_angle = steering_angle

        
class VehicleState:
    def __init__(self, shape=Rectangle(length=0, width=0), Type=None):
        self.velocity = 0
        self.position = np.array([0.0, 0.0])
        self.orientation = 0
        self.ID = 0
        self.shape = shape
        self.type = Type
        
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
        return [-5, -2.5, 0.0, 2.5]
        
    def takeAction(self, action, dt = 0.5):
        state = deepcopy(self)
        
        # Update next ego state
        state.ego = self.calculateNextEgoState(state, action, dt)
        
        # Update next vehicles state
        state.vehicles = self.calculateNextVehicleState(state, dt)
        
        # Update status between ego and vehicles
        state.status = self.calculateNextStatus(state)
        
        return state
    
    def calculateNextEgoState(self, state, action, dt):
        t1 = time.time()
        if state.ego.velocity < 0:
            state.ego.velocity = 0.0
            
        # state.ego.position[0] += (state.ego.velocity * dt + 0.5 * action * dt ** 2) *  math.cos(math.radians(state.ego.orientation) / (math.pi / 2) * 90)
        # state.ego.position[1] += (state.ego.velocity * dt + 0.5 * action * dt ** 2) *  math.sin(math.radians(state.ego.orientation) / (math.pi / 2) * 90)
        state.ego.path_length += state.ego.velocity * dt + 0.5 * action * dt ** 2
        index = DataHandler().lower_bound(list_s, state.ego.path_length)
        if index >= len(_waypoints_ego):
            index = len(_waypoints_ego) - 1
        waypoint_right = _waypoints_ego[index]
        waypoint_left = _waypoints_ego[index - 1]
        weight = (state.ego.path_length - waypoint_left.path_length) / (waypoint_right.path_length - waypoint_left.path_length)
        
        state.ego.position[0] = weight * (waypoint_right.position[0] - waypoint_left.position[0]) + waypoint_left.position[0]
        state.ego.position[1] = weight * (waypoint_right.position[1] - waypoint_left.position[1]) + waypoint_left.position[1]
        state.ego.orientation = weight * (waypoint_right.orientation - waypoint_left.orientation) + waypoint_left.orientation
        state.ego.velocity += action * dt
        state.ego.time_step += 5
        
        new_ego_state = State(position=state.ego.position, velocity=state.ego.velocity, orientation=state.ego.orientation, time_step =state.ego.time_step)
        state.ego.trajectory.append(new_ego_state)
        t2 = time.time()
        # print('calculateEgo', t2 - t1)
        
        return state.ego
    
    def calculateNextVehicleState(self, state, dt):
        # Vehicles is a list of instances
        # Predict the state of vehicles
        t1 = time.time()
        sorted_vehicles = DataHandler().sort_vehicle(state.ego, state.vehicles)
        state.vehicles = []
        for i in range(len(sorted_vehicles)):
            state.vehicles.append(sorted_vehicles[i][0])
        
        if len(state.vehicles) != 0:
            for i, vehicle in enumerate(state.vehicles):
                if vehicle.type == 'dynamic':
                    if i <= 1:
                        vehicle.path_length = (np.array(vehicle.path_length) + vehicle.velocity * dt).tolist()
                        random_i = random.randint(0, len(list_s_v[vehicle.ID]) - 1) 
                        index = DataHandler().lower_bound(list_s_v[vehicle.ID][random_i], vehicle.path_length[random_i])
                        if index >= len(_waypoints_vehicles[vehicle.ID][random_i]):
                            index = len(_waypoints_vehicles[vehicle.ID][random_i]) - 1
                        waypoint_right = _waypoints_vehicles[vehicle.ID][random_i][index]
                        waypoint_left = _waypoints_vehicles[vehicle.ID][random_i][index - 1]
                        weight = (vehicle.path_length[random_i] - waypoint_left.path_length) / (waypoint_right.path_length - waypoint_left.path_length)
                        vehicle.position[0] = weight * (waypoint_right.position[0] - waypoint_left.position[0]) + waypoint_left.position[0]
                        vehicle.position[1] = weight * (waypoint_right.position[1] - waypoint_left.position[1]) + waypoint_left.position[1]
                        vehicle.orientation = weight * (waypoint_right.orientation - waypoint_left.orientation) + waypoint_left.orientation
                    else:
                        vehicle.position[0] += vehicle.velocity * dt * math.cos(math.radians(vehicle.orientation) / (math.pi / 2) * 90)
                        vehicle.position[1] += vehicle.velocity * dt * math.sin(math.radians(vehicle.orientation) / (math.pi / 2) * 90)                        

        t2 = time.time()
        # print('calculateVehicle', t2 - t1)
        return state.vehicles
    
    
    def calculateNextStatus(self, state):
        if state.ego.goal.is_reached(state.ego):
            state.status.is_reached = True
            
        sorted_vehicles = DataHandler().sort_vehicle(state.ego, state.vehicles)
        state.vehicles = []
        for i in range(len(sorted_vehicles)):
            state.vehicles.append(sorted_vehicles[i][0])
        
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
        t1 = time.time()
        ego_points = np.array([[state.ego.shape.length / 2, state.ego.shape.width / 2],
                               [state.ego.shape.length / 2, -state.ego.shape.width / 2],
                               [-state.ego.shape.length / 2, -state.ego.shape.width / 2],
                               [-state.ego.shape.length / 2, state.ego.shape.width / 2]])
        
        polygon_ego = self.getPolygon(ego_points, state.ego.position[0], state.ego.position[1], state.ego.orientation)
        polygon_vehicles = []
        for vehicle in state.vehicles:
            vehicle_points = np.array([[vehicle.shape.length / 2, vehicle.shape.width / 2],
                                       [vehicle.shape.length / 2, -vehicle.shape.width / 2],
                                       [-vehicle.shape.length / 2, -vehicle.shape.width / 2],
                                       [-vehicle.shape.length / 2, vehicle.shape.width / 2]])
            polygon_vehicle = self.getPolygon(vehicle_points, vehicle.position[0], vehicle.position[1], vehicle.orientation)
            polygon_vehicles.append(polygon_vehicle)
            
        for polygon_vehicle in polygon_vehicles:
            if polygon_ego.intersects(polygon_vehicle):
                state.status.is_collided = True
                break
        
        
        
        """
        3. Use the collision checker that comes with commonroad
        
        cc = create_collision_checker(scenario) 
        ego_trajectory = Trajectory(0, state.ego.trajectory)
        
        # create a TrajectoryPrediction object consisting of the trajectory and the shape of the ego vehicle
        traj_pred = TrajectoryPrediction(trajectory=ego_trajectory, shape=state.ego.shape)
        co = create_collision_object(traj_pred)
        state.status.is_collided = cc.collide(co)
        """
        
        """
        4. Three-circle decomposition
        
        circle_ego = self.getThreeCircle(state.ego.position[0], state.ego.position[1], state.ego.orientation, state.ego.shape.length, state.ego.shape.width)
        circle_vehicles = []
        for vehicle in state.vehicles:
            circle_vehicle = self.getThreeCircle(vehicle.position[0], vehicle.position[1], vehicle.orientation, vehicle.shape.length, vehicle.shape.width)
            circle_vehicles.append(circle_vehicle)
            
        for circle_vehicle in circle_vehicles:
            if circle_ego.intersects(circle_vehicle):
                # How to prevent lateral errors ?
                state.status.is_collided = True
                break
        """  
        t2 = time.time()
        # print('collisionchecker', t2 - t1)
        return state.status
    
    def getPolygon(self, vehicle_points, x_tra, y_tra, orientation):
        xy_list_rot_tra = []
        for i, vehicle_points in enumerate(vehicle_points):
            x_rot = vehicle_points[0] * math.cos(math.radians(orientation / (math.pi / 2) * 90)) - vehicle_points[1] * math.sin(math.radians(orientation / (math.pi / 2) * 90))
            y_rot = vehicle_points[0] * math.sin(math.radians(orientation / (math.pi / 2) * 90)) + vehicle_points[1] * math.cos(math.radians(orientation / (math.pi / 2) * 90))
            
            xy_list_rot_tra.append((x_rot + x_tra, y_rot + y_tra))
            
        return Polygon(xy_list_rot_tra)
    
    def getThreeCircle(self, x_tra, y_tra, orientation, length, width):
        radius = math.sqrt(length**2 / 9 + width**2 / 4)
        distance = 2 * math.sqrt(radius**2 - width**2 / 4)
        circle1 = Point(x_tra + distance * math.cos(math.radians(orientation) / (math.pi / 2) * 90), 
                        y_tra + math.sin(math.radians(orientation) / (math.pi / 2) * 90)).buffer(radius)
        circle2 = Point(x_tra, y_tra).buffer(radius)
        circle3 = Point(x_tra - distance * math.cos(math.radians(orientation) / (math.pi / 2) * 90),
                        y_tra - math.sin(math.radians(orientation) / (math.pi / 2) * 90)).buffer(radius)
        
        circle_union = cascaded_union([circle1, circle2, circle3])
        return circle_union
    
    def isTerminal(self):
        return self.status.is_reached or self.status.is_collided
    
    def getReward(self, action=None):
        ego_speed = self.ego.velocity
        # if collision, reward -2000-v^2; if success, reward 1000
        if self.isTerminal():
            return -2000 - ego_speed ** 2 if self.status.is_collided else 1000
        
        # speed reward: 0 reward when 8.33 ~ 16.67 m/s(30 ~ 60 km/h); out of the range negative reward        
        speed_reward = 5.0 * (ego_speed - 8.33) if ego_speed < 8.33 else (
            0 if ego_speed < 16.67 else 4 * (16.67 - ego_speed))
        # action reward
        action_reward = -0.1 if action != 0 else 0
      
        return speed_reward + action_reward


    
"""
exp3-algorithm
"""
    
    
def draw(probs):
    z = random.random()
    cum_prob = 0.0
    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob
        if cum_prob > z:
            return i

    return len(probs) - 1

class Exp3():
    def __init__(self, gamma, n_arms):
        self.gamma = gamma
        self.weights = [1.0 for i in range(n_arms)]
        

    def select_arm(self):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = [0.0 for i in range(n_arms)]
        for arm in range(n_arms):
            probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
            probs[arm] = probs[arm] + (self.gamma) * (1.0 / float(n_arms))
        return draw(probs)

    def update(self, chosen_arm, reward):
        n_arms = len(self.weights)
        total_weight = sum(self.weights)
        probs = [0.0 for i in range(n_arms)]
        for arm in range(n_arms):
            probs[arm] = (1 - self.gamma) * (self.weights[arm] / total_weight)
            probs[arm] = probs[arm] + (self.gamma) * (1.0 / float(n_arms))

        x = reward / probs[chosen_arm]

        growth_factor = math.exp((self.gamma / n_arms) * x)
        self.weights[chosen_arm] = self.weights[chosen_arm] * growth_factor 
       

"""
mcts
"""

strategyset = [0.001, 1, 1000]
# ind = 0
normalized_score = [0.0000001]
exp3 = Exp3(0.5, len(strategyset))

def randomPolicy(state):
    accum_reward = 0
    for i in range(6):
        if not state.isTerminal():
            action = random.choice(state.getPossibleActions())
            state = state.takeAction(action)
            accum_reward += state.getReward(action)
            # print(action, state.getReward(action), accum_reward)
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
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1,
                 rolloutPolicy=randomPolicy, index=0):
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
        self.index = index

    def search(self, initialState, needDetails=False):
        self.root = treeNode(initialState, None)
        
        t1 = time.time()
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()
        

        bestChild = self.getBestChild(self.root, self.explorationConstant)
        action = (action for action, node in self.root.children.items() if node is bestChild).__next__()
        t2 = time.time()
        print('Search for next state time', t2 - t1)
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
        score = []
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, strategyset[self.index])
                
                if not node.isFullyExpanded:
                    score.append(node.totalReward / node.numVisits)
                    normalized_score.append(np.mean(score))
                    exp3.update(self.index, (np.mean(score) - min(normalized_score)) / (max(normalized_score) - min(normalized_score)))
                    print(exp3.weights)
                    self.index = exp3.select_arm()
                
            else:
                # print(node, node.children)
                return self.expand(node)
        
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                # print(node, node.children)
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
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        # print(bestNodes)
        return random.choice(bestNodes)     


class CommonroadEnv():
    def __init__(self, scenario, planning_problem_set):
        self.scenario = scenario
        self.planning_problem_set = planning_problem_set
        self.state = StateSpace()
        # self.time_step = 0
        self.episode = -1
        self.dt = 0.5
        self.reward = 0
        self.done = False
        self.action = 0
        self.ego_initial_state = self.generate_ego_initial_state()
        self.vehicles_initial_state = self.generate_vehicles_initial_state()
        # self.trajectory_state_list = [State(position = self.ego_initial_state.position, velocity = self.ego_initial_state.velocity, orientation = self.ego_initial_state.orientation, time_step = self.time_step)]
        self.df_ego = pd.DataFrame(columns=['time_step', 'velocity', 'position', 'orientation', 'action', 'reward'])
        
    def generate_df(self, df):
        state = deepcopy(self.state)
        df = df.append(pd.Series({'time_step': state.ego.time_step, 
                                  'velocity':state.ego.velocity, 
                                  'position': state.ego.position, 
                                  'orientation': state.ego.orientation, 
                                  'action': self.action,
                                  'reward': self.reward}, name = self.episode))

        return df
        
    def generate_ego_initial_state(self):
        for key in self.planning_problem_set.planning_problem_dict.keys():
            # self.goal = planning_problem_set.planning_problem_dict[key].goal
            ego_initial_state = EgoState(velocity=self.planning_problem_set.planning_problem_dict[key].initial_state.velocity,
                                         position=self.planning_problem_set.planning_problem_dict[key].initial_state.position, 
                                         orientation=self.planning_problem_set.planning_problem_dict[key].initial_state.orientation, 
                                         time_step=self.planning_problem_set.planning_problem_dict[key].initial_state.time_step,
                                         goal=self.planning_problem_set.planning_problem_dict[key].goal,
                                         trajectory=[State(position=self.planning_problem_set.planning_problem_dict[key].initial_state.position, velocity=self.planning_problem_set.planning_problem_dict[key].initial_state.velocity, orientation=self.planning_problem_set.planning_problem_dict[key].initial_state.orientation, time_step=self.planning_problem_set.planning_problem_dict[key].initial_state.time_step, steering_angle=0)])
            
        return ego_initial_state
    
    def generate_vehicles_initial_state(self):
        vehicles_initial_state = []
        
        for i in range(len(self.scenario.dynamic_obstacles)):
            vehicle = VehicleState(Type='dynamic')
            vehicle.position = self.scenario.dynamic_obstacles[i].initial_state.position
            vehicle.orientation = self.scenario.dynamic_obstacles[i].initial_state.orientation
            vehicle.velocity = self.scenario.dynamic_obstacles[i].initial_state.velocity
            vehicle.ID = self.scenario.dynamic_obstacles[i].obstacle_id
            vehicle.shape = Rectangle(length=self.scenario.dynamic_obstacles[i].obstacle_shape.length, width=self.scenario.dynamic_obstacles[i].obstacle_shape.width)
            l = []
            for j in range(len(dic[self.scenario.dynamic_obstacles[i].obstacle_id])):
                l.append(dic[self.scenario.dynamic_obstacles[i].obstacle_id][j][0])
                vehicle.path_length = l
            vehicles_initial_state.append(vehicle)
            
        for i in range(len(self.scenario.static_obstacles)):
            vehicle = VehicleState(Type='static')
            vehicle.position = self.scenario.static_obstacles[i].initial_state.position
            vehicle.orientation = self.scenario.static_obstacles[i].initial_state.orientation
            vehicle.velocity = self.scenario.static_obstacles[i].initial_state.velocity
            vehicle.ID = self.scenario.static_obstacles[i].obstacle_id
            vehicle.shape = Rectangle(length=self.scenario.static_obstacles[i].obstacle_shape.length, width=self.scenario.static_obstacles[i].obstacle_shape.width)
            vehicle.path_length = 0
            vehicles_initial_state.append(vehicle)
            
        return vehicles_initial_state
        
    
    def get_current_vehicle_waypoint(self, vehicle):
        # waypoints = deepcopy(_waypoints_vehicles)
        waypoints = DataHandler().choose_vehicle_route(_waypoints_vehicles[vehicle.ID])

        return waypoints[_distance.index(min(_distance))], _distance.index(min(_distance))
    
    def start(self):
        self.state.ego = self.ego_initial_state
        self.state.vehicles = self.vehicles_initial_state
        
        """
        
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
            
        """
        
        return self.state
    
    def step(self, action):
        self.action = action
        
        """
        Bring the speed or action of the ego into the sumo simulator
        
        """

        # self.state.ego.position[0] += (self.state.ego.velocity * self.dt + 0.5 * action * self.dt ** 2) * math.cos(math.radians(self.state.ego.orientation) / (math.pi / 2) * 90)
        # self.state.ego.position[1] += (self.state.ego.velocity * self.dt + 0.5 * action * self.dt ** 2) * math.sin(math.radians(self.state.ego.orientation) / (math.pi / 2) * 90)
        self.state.ego.path_length += self.state.ego.velocity * self.dt + 0.5 * action * self.dt ** 2
        index = DataHandler().lower_bound(list_s, self.state.ego.path_length)
        if index < len( _waypoints_ego):
            
            waypoint_right = _waypoints_ego[index]
            waypoint_left = _waypoints_ego[index - 1]
            weight = (self.state.ego.path_length - waypoint_left.path_length) / (waypoint_right.path_length - waypoint_left.path_length)

            self.state.ego.position[0] = weight * (waypoint_right.position[0] - waypoint_left.position[0]) + waypoint_left.position[0]
            self.state.ego.position[1] = weight * (waypoint_right.position[1] - waypoint_left.position[1]) + waypoint_left.position[1]
            self.state.ego.orientation = weight * (waypoint_right.orientation - waypoint_left.orientation) + waypoint_left.orientation
            self.state.ego.velocity += action * self.dt
            curvature_s = weight * (waypoint_right.curvature - waypoint_left.curvature) + waypoint_left.curvature
            self.state.ego.steering_angle = np.arctan(curvature_s * self.state.ego.shape.length)

            if self.state.ego.velocity < 0:
                self.state.ego.velocity = 0.0
            self.state.ego.time_step = self.state.ego.time_step + 5

            for vehicle in self.state.vehicles:
                if vehicle.type == 'dynamic':
                    vehicle.position = self.get_obstacle_state_at_time()['dynamic'][vehicle.ID].position
                    vehicle.orientation = self.get_obstacle_state_at_time()['dynamic'][vehicle.ID].orientation
                    vehicle.velocity = self.get_obstacle_state_at_time()['dynamic'][vehicle.ID].velocity
                    vehicle.path_length = (np.array(vehicle.path_length) + vehicle.velocity * self.dt).tolist()

            new_ego_state = State(position = self.state.ego.position, velocity = self.state.ego.velocity, orientation = self.state.ego.orientation, time_step = self.state.ego.time_step, steering_angle=self.state.ego.steering_angle)
            self.state.ego.trajectory.append(deepcopy(new_ego_state))
            self.done = self.is_done()
            self.reward = self.get_reward()
            self.df_ego = self.generate_df(self.df_ego)
            
        else:
            self.done = True
            
        return self.state, self.reward, self.done
    
    def reset(self):
        # self.state.ego.time_step = 0
        self.action = 0
        self.reward = 0
        self.done = False
        self.state.ego = deepcopy(self.ego_initial_state)
        self.state.vehicles = deepcopy(self.vehicles_initial_state)
        self.state.ego.trajectory = [State(position = self.ego_initial_state.position, velocity = self.ego_initial_state.velocity, orientation = self.ego_initial_state.orientation, time_step = self.state.ego.time_step, steering_angle=0)]
        self.episode += 1
        self.df_ego = self.generate_df(self.df_ego)
        
        return self.state
        
    def get_reward(self):
        # if collision, reward -2000-v^2; if success, reward 1000
        if self.done:
            if self.is_collided():
                self.reward = -2000 - self.state.ego.velocity ** 2
            elif self.time_run_out():
                self.reward = -500
            else:
                self.reward = 4000
                
            return self.reward
                
        # speed reward: 0 reward when 8.33 ~ 16.67 m/s(30 ~ 60 km/h); out of the range negative reward
        
        speed_reward = 5.0 * (self.state.ego.velocity - 8.33) if self.state.ego.velocity < 8.33 else (
            0 if self.state.ego.velocity < 16.67 else 4 * (16.67 - self.state.ego.velocity))
        # action reward
        action_reward = -0.1 if self.action != 0 else 0
    
        self.reward = speed_reward + action_reward
        return self.reward
    
    def is_done(self):
        return self.is_reached() or self.is_collided() or self.time_run_out()
    
    def is_reached(self):
        return self.ego_initial_state.goal.is_reached(self.state.ego)
    
    def is_collided(self):
        cc = create_collision_checker(scenario) 
        # ego_trajectory = Trajectory(0, self.state.ego.trajectory)
        ego_trajectory = Trajectory(self.state.ego.time_step, [State(time_step=self.state.ego.time_step, position=self.state.ego.position, orientation=self.state.ego.orientation)])
        
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
    
    def time_run_out(self):
        if self.state.ego.time_step > 99:
            return True
        
    def get_obstacle_state_at_time(self):
        # Returns a dictionary containing static and dynamic obstacles
        obstacle_states = {'dynamic':{}, 'static':{}}

        for obstacle in scenario.dynamic_obstacles:
            if obstacle.state_at_time(self.state.ego.time_step) is not None:
                obstacle_states['dynamic'][obstacle.obstacle_id] = obstacle.state_at_time(self.state.ego.time_step)
            else:
                obstacle_states['dynamic'][obstacle.obstacle_id] =  State(position=np.array([-999,-999]), 
                                                                          orientation=0, 
                                                                          velocity=0, 
                                                                          acceleration=0,
                                                                          time_step=self.state.ego.time_step)
        for obstacle in scenario.static_obstacles:
            obstacle_states['static'][obstacle.obstacle_id] = obstacle.initial_state

        return obstacle_states
    
    def visualization(self):
        return visualize_solution(self.scenario, self.planning_problem_set, Trajectory(0, self.state.ego.trajectory))


class DataHandler():
    def __init__(self, number_vehicle_to_handle=5, method_sort_vehicle='nearst_vehicle', method_choose_route='random'):
        self.number_vehicle_to_handle = number_vehicle_to_handle
        self.method_sort_vehicle = method_sort_vehicle
        self.method_choose_route = method_choose_route
        
    def sort_vehicle(self, ego: EgoState, vehicles: VehicleState) -> VehicleState:
        # Sorted the several important vehicles by such as distance between ego and vehicle
        if self.method_sort_vehicle == 'nearst_vehicle':
            considered_vehicles = [[vehicle, distance.euclidean(ego.position, vehicle.position)] for vehicle in vehicles 
                                   if distance.euclidean(ego.position, vehicle.position) < 30]
            considered_vehicles = sorted(considered_vehicles, key=(lambda x: x[1]))[0:self.number_vehicle_to_handle]
            return considered_vehicles
    
    def choose_vehicle_route(self, route_waypoint):
        if self.method_choose_route == 'random':
            return route_waypoint[random.randint(0, len(route_waypoint) - 1)]
        
        elif self.method_choose_route == 'sample-based':
            # needs to be modified
            epsilon = 0.92
            p = np.ones(len(route_waypoint)) * epsilon / len(route_waypoint)
            reward = np.zeros(len(route_waypoint))
            worst_route = np.argmax(reward)
            p[worst_route] = 1 - epsilon + (epsilon / len(route_waypoint))
            return route_waypoint[np.random.choice(np.arange(len(route_waypoint)), p=p)]
    
    def lower_bound(self, nums, target):
        low, high = 0, len(nums)-1
        pos = len(nums)
        while low<high:
            mid = (low+high)//2
            if nums[mid] < target:
                low = mid+1
            else:
                high = mid

        if nums[low]>=target:
            pos = low

        return pos


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
    scores_window = deque(maxlen=1)
    actions = [-5, -2.5, 0.0, 2.5]
    df_evaluation = pd.DataFrame(columns=['success_rate', 'steps', 'collision_speed', 'score', 'runtime'])


    # run episode
    for episode in episodes:
        score = 0
        state = env.reset()
        searcher = mcts(iterationLimit=100)
        ind = 0
        t1 = time.time()
        while True:
            mcts_state = state
            action = searcher.search(initialState=mcts_state)
            state, reward, done = env.step(action)
            print(reward, env.is_collided(), env.state.ego.velocity, env.state.ego.time_step, env.state.ego.steering_angle)
            score += reward

            if done:
                t2 = time.time()
                
                if env.is_reached():
                    success = 1
                        
                else:
                    success = 0
                df_evaluation = df_evaluation.append(pd.Series({'success_rate': success, 
                            'steps':env.state.ego.time_step, 
                            'collision_speed': state.ego.velocity if env.is_collided() else np.nan, 
                            'score': score, 
                            'runtime': t2 - t1}, name = episode))
                scores.append(score)
                average_scores.append(sum(scores) / (episode + 1))
                print("episode: {}, score: {:.2f}".format(episode, score))
                # env.visualization()
                break


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(scores)), scores, label = 'score')
    ax.plot(np.arange(len(scores)), average_scores, label = 'average score')
    ax.legend()
    plt.ylabel('score')
    plt.xlabel('Episode')

    # generate png
    dir_path_png = os.path.join(os.getcwd(), 'score')
    if not os.path.isdir(dir_path_png):
        os.makedirs(dir_path_png)
    plt.savefig(dir_path_png + '/' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.png')
    plt.show()

    # generate csv for ego information
    dir_path_csv = os.path.join(os.getcwd(), 'csv_ego')
    if not os.path.isdir(dir_path_csv):
        os.makedirs(dir_path_csv)
    env.df_ego.to_csv(dir_path_csv + '/' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.csv', index = True, sep = ' ', float_format='%.4f')

    # generate csv for evaluation
    dir_path_csv = os.path.join(os.getcwd(), 'csv_eva')
    if not os.path.isdir(dir_path_csv):
        os.makedirs(dir_path_csv)
    df_evaluation.to_csv(dir_path_csv + '/' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.csv', index = True, sep = ' ', float_format='%.4f')

