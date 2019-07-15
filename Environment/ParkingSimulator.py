# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version Copyright (c) 2010 kne / sirkne at gmail dot com
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

# This code modifies the Box2D package from the OpenAI Gym to build an
# environment for testing automated parking algorithms.


import math
import numpy as np
import time

from .car_dynamics import Car
import gym
from gym import spaces
from gym.utils import seeding
import copy
import itertools

from Box2D import b2World, b2FixtureDef, b2PolygonShape

# This is set very low. May not look good when
# rendered but makes learning easier
FPS = 5
# Affects how fast-paced the game is, forces should be adjusted as well
SCALE = 10.0

# Size variables for the car length and width,
# and relative spot length and width
car_length = 250 * 0.02
car_width = 120 * 0.02
spot_length = 1.5 * car_length
spot_width = 2 * car_width

VIEWPORT_W = 1200
VIEWPORT_H = 800

PARKED_REWARD = 1


class ParkingSimulator(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    special_angle_lim = 0.03
    v_lim = (0.15, 0.85)
    h_lim = (0.15, 0.85)
    min_speed = 0
    max_speed = 2

    def __init__(self, rand_spot=True,
                 sim_mode='continuous', special_eps=False, state_type='summary',
                 render_ep=False, gamma=1, timestep_as_channel=False, reset_ep=1e5):
        self._seed()
        self.viewer = None
        self.render_ep = render_ep

        self.world = b2World()
        self.parkinglot = None
        self.spot = None
        self.car = None

        self.out_of_bounds = False
        self.parked = False
        self.in_spot = False
        self.corner_in_spot = False

        self.reward = 0.0
        self.prev_reward = 0.0

        self.best_dist = np.infty

        # Dimensions of the window. Logest_dist is
        # the angular distance from corner
        # to corner. An upper bound on how far away
        # the car can be from the spot.
        self.W = VIEWPORT_W / SCALE
        self.H = VIEWPORT_H / SCALE
        #self.longest_dist = np.sqrt(((self.h_lim[1] - self.h_lim[0]) * self.W) ** 2 +
        #                            ((self.v_lim[1] - self.v_lim[0]) * self.H) ** 2)
        self.longest_dist = np.sqrt(self.W ** 2 + self.H ** 2)

        self.rand_spot = rand_spot
        self.special_eps = special_eps
        if sim_mode in ['continuous', 'discrete']:
            self.mode = sim_mode
        else:
            raise RuntimeError('"{}" is not a valid simulator mode'.format(sim_mode))

        assert state_type in ['summary', 'visual']
        self.state_type = state_type

        self.gamma = gamma

        if self.state_type == 'summary':
            # Speed
            # Hull Angle (Heading)
            # Steering Angle
            # Distance to Parking Spot
            # Angle to Parking Spot
            # X-Y Coords of Parking Spot Corners (4 values)
            # X-Y Coords of Car Corners (16 Values)
            low = np.array([-0.01, -1.01, -1.01, -0.01, -1.01] +
                           [-0.01] * 4 +
                           [-0.01] * 16)
            high = np.array([1.01] * 25)

            # self.observation_space = spaces.Box(low=low,
            # high=high,name='observation_space', dtype=np.dtype(float).type)
            self.observation_space = spaces.Box(low, high, dtype=np.float32)
            self.timestep_as_channel = None

        elif self.state_type == 'visual':
            self.timestep_as_channel = timestep_as_channel
            if timestep_as_channel:
                dummy_state = np.ones([VIEWPORT_H, VIEWPORT_W, 12])
            else:
                dummy_state = np.ones([4, VIEWPORT_H, VIEWPORT_W, 3])

            self.observation_space = spaces.Box(0 * dummy_state, dummy_state, dtype=np.float32)
            self.state = dummy_state

        if self.mode == 'continuous':
            # Action is two floats [Acceleration, Steering rate].
            # Acceleration: velocity changes proportionally to acceleration
            # Steering-rate:  steering angle changes proportionally to steering rate
            self.action_space = spaces.Box(
                np.array([-1.0, -1.0]), np.array([+1.0, +1.0]),
                dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(9)
            low_acc = -0.5
            mid_acc = 0
            high_acc = 1
            low_steer = -0.5
            mid_steer = 0.0
            high_steer = 0.5
            self.action_mapping = {'acceleration': [low_acc, low_acc, low_acc,
                                                    mid_acc, mid_acc, mid_acc,
                                                    high_acc, high_acc, high_acc],
                                   'steering': [low_steer, mid_steer, high_steer,
                                                low_steer, mid_steer, high_steer,
                                                low_steer, mid_steer, high_steer]}

        # Create the car with starting conditions to be modified in _reset
        self.car = Car(self.world, 0,
                       self.W / 2, self.H / 2, self.max_speed)

        self.reset_ep = reset_ep
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.parkinglot:
            return
        self.world.DestroyBody(self.grass)
        self.grass = None
        self.world.DestroyBody(self.car)
        self.car = None
        self.world = None

        self.lot_polys = None
        self.grass_polys = None

    def reset(self):
        return self._reset(self.reset_ep)

    def _reset(self, episode):
        self._destroy()
        self.game_over = False

        self.parked = False
        self.in_spot = False
        self.corner_in_spot = False
        self.ever_in_spot = False
        self.ever_cis = [False] * len(self.car.corners)
        self.first_cis = [False] * len(self.car.corners)
        self.first_spot_entry = False

        self.reward = 0
        self.prev_reward = 0

        # Tracking Episode Stats
        self.ep_message = []
        self.ep_reward = 0
        self.ep_len = 0

        # Spot can be either horizontally oriented or vertically
        # If rand_spot, randomly select which it will be.
        if self.rand_spot:
            self.spot_orient = np.int(self.np_random.uniform(0, 2, size=1)[0])

            if self.spot_orient == 0:  # horizontal spot
                self.spot_x1 = self.np_random.uniform(
                    self.h_lim[0] * self.W, self.h_lim[1] * self.W - spot_length)
                self.spot_x2 = self.spot_x1 + spot_length

                self.spot_y1 = self.np_random.uniform(
                    self.v_lim[0] * self.H, self.v_lim[1] * self.H - spot_width)
                self.spot_y2 = self.spot_y1 + spot_width

            else:  # vertical spot
                self.spot_x1 = self.np_random.uniform(
                    self.h_lim[0] * self.W, self.h_lim[1] * self.W - spot_width)
                self.spot_x2 = self.spot_x1 + spot_width

                self.spot_y1 = self.np_random.uniform(
                    self.v_lim[0] * self.H, self.v_lim[1] * self.H - spot_length)
                self.spot_y2 = self.spot_y1 + spot_length

        else:  # spot in center of lot
            self.spot_orient = 0

            self.spot_x1 = 0.5 * (self.W - spot_length)
            self.spot_x2 = self.spot_x1 + spot_length

            self.spot_y1 = 0.5 * (self.H - spot_width)
            self.spot_y2 = self.spot_y1 + spot_width

        self.spot_corners = [self.spot_x1, self.spot_x2, self.spot_y1, self.spot_y2]

        # Define spaces of where the two parts of the lot are so that later
        # this can be checked to establish terminal conditions.
        self.lot_space = spaces.Box(
            np.array([self.h_lim[0] * self.W, self.v_lim[0] * self.H]),
            np.array([self.h_lim[1] * self.W, self.v_lim[1] * self.H]),
            dtype=np.float32)
        self.spot_space = spaces.Box(
            np.array([self.spot_x1, self.spot_y2]),
            np.array([self.spot_x2, self.spot_y1]),
            dtype=np.float32)

        # Center of the parking spot, to be passed as a state variable
        self.spot_x_loc = 0.5 * (self.spot_x1 + self.spot_x2)
        self.spot_y_loc = 0.5 * (self.spot_y1 + self.spot_y2)

        # Creating shapes to draw parking lot and spot
        # Defines the vertices of the parking lot border
        '''
        self.lot_poly = [(self.h_lim[0] * self.W, self.v_lim[0] * self.H),
                         (self.h_lim[0] * self.W, self.v_lim[1] * self.H),
                         (self.h_lim[1] * self.W, self.v_lim[1] * self.H),
                         (self.h_lim[1] * self.W, self.v_lim[0] * self.H)]
        '''
        self.lot_polys = [
            [(self.h_lim[0] * self.W, self.v_lim[0] * self.H),
             (self.spot_x1, self.spot_y1),
             (self.spot_x2, self.spot_y1),
             (self.h_lim[1] * self.W, self.v_lim[0] * self.H)],
            [(self.h_lim[1] * self.W, self.v_lim[0] * self.H),
             (self.spot_x2, self.spot_y1),
             (self.spot_x2, self.spot_y2),
             (self.h_lim[1] * self.W, self.v_lim[1] * self.H)],
            [(self.h_lim[1] * self.W, self.v_lim[1] * self.H),
             (self.spot_x2, self.spot_y2),
             (self.spot_x1, self.spot_y2),
             (self.h_lim[0] * self.W, self.v_lim[1] * self.H)],
            [(self.h_lim[0] * self.W, self.v_lim[1] * self.H),
             (self.spot_x1, self.spot_y2),
             (self.spot_x1, self.spot_y1),
             (self.h_lim[0] * self.W, self.v_lim[0] * self.H)]]

        self.spot_poly = [(self.spot_x1, self.spot_y2),
                          (self.spot_x2, self.spot_y2),
                          (self.spot_x2, self.spot_y1),
                          (self.spot_x1, self.spot_y1)]

        # Need to make solid grass areas that don't
        # cover the lot and parking spot
        self.grass_polys = [
            [(0, 0),
             (self.h_lim[0] * self.W, self.v_lim[0] * self.H),
             (self.h_lim[1] * self.W, self.v_lim[0] * self.H),
             (self.W, 0)],
            [(self.W, 0),
             (self.h_lim[1] * self.W, self.v_lim[0] * self.H),
             (self.h_lim[1] * self.W, self.v_lim[1] * self.H),
             (self.W, self.H)],
            [(0, self.H),
             (self.h_lim[0] * self.W, self.v_lim[1] * self.H),
             (self.h_lim[1] * self.W, self.v_lim[1] * self.H),
             (self.W, self.H)],
            [(0, self.H),
             (self.h_lim[0] * self.W, self.v_lim[1] * self.H),
             (self.h_lim[0] * self.W, self.v_lim[0] * self.H),
             (0, 0)]]

        self.world = b2World()
        self.grass = self.world.CreateStaticBody(
            fixtures=[b2FixtureDef(
                shape=b2PolygonShape(
                    vertices=[x for x in poly]),
                density=1.0) for poly in self.grass_polys])
        self.drawlist = [self.grass]

        self.place_car()

        # Some tracking variables for rewards
        # Variable stored for calculating progress towards spot
        self.old_dist = self.get_dist_to_spot()
        # Variable can be used to only reward if the car get's closer than previous best
        self.best_dist = self.get_dist_to_spot()
        self.start_dist = self.get_dist_to_spot()
        # Can be used to reward smooth actions
        self.last_action = [0] * 1

        # Return the starting state
        self.get_state()

        return self.state

    def place_car(self):
        # Creating the car and placing it randomly
        # For now I am leaving special episodes just commented out, but may remove entirely later

        initial_x = self.np_random.uniform((self.h_lim[0] + 0.2) * self.W,
                                           (self.h_lim[1] - 0.2) * self.W, 1)[0]
        initial_y = self.np_random.uniform((self.v_lim[0] + 0.2) * self.H,
                                           (self.v_lim[1] - 0.2) * self.H, 1)[0]
        # initial_heading = self.np_random.uniform(-math.pi, math.pi, 1)[0]
        initial_heading = self.compute_angle_to_spot()
        initial_speed = self.np_random.uniform(0, self.max_speed / 2, 1)[0]

        self.car = Car(self.world, initial_heading, initial_x, initial_y, initial_speed)

        '''
        # A defined fraction of episodes are randomly selected as 'special episodes',
        # where the car begins very close, facing directly at the spot, and has little chance
        # of missing it. This is to encourage convergence and provide positive examples in the
        # history.
        U = self.np_random.uniform(0, 1, 1)[0]
        special_ep_prob = np.maximum(0.4 - 0.1 * episode / 1000, 0.3)
        if U < special_ep_prob and self.special_eps:
            self.ep_message.append('Special Episode')

            self.special_ep = True

            dist_lim = self.np_random.uniform(5, np.minimum(15 + 5 * episode / 1000, 45), 1)[0]
            angle_lim = self.special_angle_lim * 1.3

            # Don't perturb the angle on special episodes
            angle_perturb = 0

        else:
            self.ep_message.append('Normal Episode')
            self.special_ep = False

            # Set the maximum starting distance and maximum starting angle (larger is harder)
            # from the spot. Increases with experience.
            dist_lim = np.minimum(10 + 1 * episode / 1000, 45)
            angle_lim = np.minimum(
                (0.01 + 0.005 * episode / 1000) * math.pi , math.pi / 4)

            # Randomly turn the car so it doesn't always face directly at the spot.
            # This is very minimal
            angle_perturb = self.np_random.normal(0,
                                                  (0.001 +
                                                   0.005 * episode / 1000),
                                                  1)[0]
        if self.spot_side == 0:
            self.ref_angle = -math.pi
        elif self.spot_side == 1:
            self.ref_angle = -math.pi / 2
        elif self.spot_side == 2:
            self.ref_angle = 0
        elif self.spot_side == 3:
            self.ref_angle = math.pi / 2

        placed = False
        tries = 0
        while not placed:
            theta = self.np_random.uniform(low=-angle_lim,
                                           high=angle_lim,
                                           size=1)[0]
            self.ep_theta = copy.copy(theta)
            theta += self.ref_angle - math.pi / 2

            dist = self.np_random.uniform(low=5,
                                          high=dist_lim,
                                          size=1)[0]
            self.start_dist = copy.copy(dist)

            x = self.spot_x_loc + dist * math.cos(theta)
            y = self.spot_y_loc + dist * math.sin(theta)

            tries += 1

            if ((np.abs(self.compute_angle_to_spot(
                    x, y) - self.ref_angle) <= angle_lim or
                    np.abs(self.compute_angle_to_spot(
                           x, y) - self.ref_angle - 2 * math.pi) <= angle_lim) and
                (((x > (self.h_lim[0] + 0.05) * self.W and x < (self.h_lim[1] - 0.05) * self.W) and
                 (y > (self.v_lim[0] + 0.05) * self.H and y < (self.v_lim[1] - 0.05) * self.H))) or
                    dist < 10):

                placed = True
                initial_x = x
                initial_y = y
                initial_angle = self.compute_angle_to_spot(initial_x, initial_y)

            if tries > 500 and tries % 5000 == 0:
                        print('Side: {}\nx_diff: {}\ny_diff: {}\nAngle to spot: {}'.format(
                            self.spot_side,
                            np.round(x - self.spot_x_loc, 2),
                            np.round(y - self.spot_y_loc, 2),
                            np.round(dist, 2),
                            np.round(np.abs(self.compute_angle_to_spot(x, y) -
                                            self.ref_angle), 2)))

        initial_angle += angle_perturb

        # This may need to be changed though
        initial_speed = self.max_speed / 2

        # self.car._reset(initial_x, initial_y, initial_angle, initial_speed)
        self.car = Car(self.world, initial_angle, initial_x, initial_y, initial_speed)
        '''

    def compute_angle_to_spot(self, x=None, y=None):
        # This function finds the angle the car needs to travel at to reach the spot
        # Calculation depends on which quadrant the spot is in

        if x is None:
            x = self.car.location[0]
        if y is None:
            y = self.car.location[1]

        if self.spot_x_loc - x == 0:
            if self.spot_y_loc - y > 0:
                est_angle = math.pi / 2
            else:
                est_angle = -math.pi / 2
        else:
            est_angle = math.atan((self.spot_y_loc - y) /
                                  (self.spot_x_loc - x))
        est_angle -= math.pi / 2
        if self.spot_x_loc - x < 0:
            if self.spot_y_loc - y > 0:
                est_angle += math.pi
            else:
                est_angle -= math.pi

        # Bound angle to between -pi and pi
        if est_angle > math.pi:
            est_angle -= 2 * math.pi
        if est_angle < -math.pi:
            est_angle += 2 * math.pi

        return est_angle

    def step(self, action):
        return self._step(action)

    def _step(self, action):

        accel, steer = self.map_action(action)

        # Functions calculate new dynamics (heading, speed, location) of car
        self.car.accelerator(accel, 1.0 / FPS)
        self.car.steering_rate(steer)
        t1, t2, t3 = self.car.step(1.0 / FPS)

        # Get the updated state
        self.get_state()

        # Checks whether car is parked, oob, partially in spot, etc and updates class values
        self.check_corners()

        done = False
        if self.out_of_bounds:
            done = True

        if self.parked:
            self.ep_message.append('Parked')
            done = True

            # If it is a special episode, the car only travels in a straight line. So we can figure
            # out the maximum angle it can succesfully park at by increasing the angle limit every
            # time the car parks at a higher angle than its previous best:
            if self.special_ep and self.ep_theta > self.special_angle_lim:
                print('Special Episode Angle '
                      'increased from {} to {}'.format(np.round(self.special_angle_lim, 2),
                                                       np.round(self.ep_theta, 2)))
                self.special_angle_lim = self.ep_theta

        # Maybe don't terminate episodes early. If car is more likely to be close to spot after
        # 500 eps then this will be seen as a punishment
        # ah well, if it's taken 500 steps and not stopped it's circling
        if self.ep_len >= 500:
            done = True
            self.ep_message.append('Stopped because more than 500 steps were taken')

        # I will instead try ending the episode if speed drops to zero
        if self.car.speed == 0 and not self.parked:
            done = True
            self.ep_message.append('Stopped because car stopped')

        step_reward = self.get_step_reward()
        self.ep_reward += (self.gamma ** self.ep_len) * step_reward
        self.ep_len += 1

        if done:
            self.ep_message.append('Reward: {}'.format(self.ep_reward))

        self.get_state()

        if done:
            info = {'episode':
                    {'r': copy.copy(self.ep_reward),
                     'l': copy.copy(self.ep_len),
                     's': copy.copy(self.parked),
                     'corner_in': copy.copy(self.corner_in_spot),
                     'e_cis': copy.copy(self.ever_cis),
                     'start_dist': copy.copy(self.start_dist),
                     'end_dist': copy.copy(self.old_dist),
                     'ep_message': self.ep_message},
                    'substep_times': [t1, t2, t3]}
        else:
            info = {'substep_times': [t1, t2, t3]}

        return self.state, step_reward, done, info

    def get_state(self):

        if self.state_type == 'summary':
            state = [self.car.speed / self.max_speed,
                     self.car.hull.angle / math.pi,
                     self.car.steering / self.car.steering_limit,
                     self.get_dist_to_spot() / self.longest_dist,
                     self.compute_angle_to_spot() / math.pi]

            state += [c / self.longest_dist for c in self.spot_corners]

            car_corners = [list(np.divide(c, self.longest_dist)) for c in self.get_corner_locs()]
            state += itertools.chain.from_iterable(car_corners)

        elif self.state_type == 'visual':
            state = np.zeros_like(self.state)
            if self.timestep_as_channel:
                state[:, :, 3:12] = state[:, :, 0:9]
                state[:, :, 0:3] = self._render(mode='rgb_array') / 255
            else:
                state[1:3, :] = self.state[0:2, :]
                state[0, :] = self._render(mode='rgb_array') / 255

        else:
            raise ValueError('{} is not a valid state_type'.format(self.state_type))

        self.state = np.array(state)

        # Check that the new state is valid wrt restrictions in __init__
        if not self.observation_space.contains(self.state):
            for d in range(len(state)):
                ib = (state[d] >= self.observation_space.low[d] and
                      state[d] <= self.observation_space.high[d])
                if not ib:
                    print('Dim {} is out of bounds'.format(d))
        assert self.observation_space.contains(self.state), "invalid state " + str(self.state)

    def render(self, mode='human', close=False):
        return self._render()

    def _render(self, mode='human', close=False):
        if not self.render_ep:
            return

        # import pdb; pdb.set_trace()
        from gym.envs.classic_control import rendering
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, self.W, 0, self.H)

        # Drawing the grass
        for p in self.grass_polys:
            # trans = f.body.transform
            # path = [trans * v for v in f.shape.vertices]
            self.viewer.draw_polygon(p, color=(0 / 255, 153 / 255, 51 / 255))

        # Drawing the pavement
        for p in self.lot_polys:
            self.viewer.draw_polygon(
                p, color=(102 / 255, 102 / 255, 102 / 255))

        # Draw the parking spot in green if car is in spot, red if it is not
        if self.in_spot:
            spot_colour = (120 / 255, 220 / 255, 150 / 255)
        else:
            spot_colour = (250 / 255, 140 / 255, 140 / 255)

        self.viewer.draw_polygon(self.spot_poly, color=spot_colour)

        # Drawing the car
        self.car.draw(self.viewer)

        # import pdb; pdb.set_trace()
        ret_state = self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return ret_state

    def get_corner_locs(self):
        # Function finds the locations of each outside corner of the car
        return [np.array([self.car.hull.GetWorldPoint(localPoint=c)[0],
                          self.car.hull.GetWorldPoint(localPoint=c)[1]])
                for c in self.car.corners]

    def check_corners(self):
        # Determining whether there is contact between car and walls
        # Find if car is in spot
        # Check if the car is at least partially in the spot

        car_corners = self.get_corner_locs()

        oob = False
        corner_in_spot = False
        in_spot = True

        self.corners_in_spot = [self.spot_space.contains(c) for c in car_corners]

        oob = not all([self.lot_space.contains(c) for c in car_corners])
        corner_in_spot = any(self.corners_in_spot)
        in_spot = all(self.corners_in_spot)

        # Update a bunch of boolean variables for the class which will influence the reward
        self.out_of_bounds = oob

        self.corner_in_spot = corner_in_spot

        self.first_cis = [(c and not e) for (e, c) in zip(self.ever_cis, self.corners_in_spot)]
        self.ever_cis = [(e or c) for (e, c) in zip(self.ever_cis, self.corners_in_spot)]

        if corner_in_spot:
            # This variable tracks if car just entered spot for first time
            if not self.ever_in_spot:
                self.first_spot_entry = True
            else:
                self.first_spot_entry = False
            self.ever_in_spot = True

        self.in_spot = in_spot
        if in_spot and self.car.speed == 0:
            self.parked = True

        return {'oob': oob, 'corner_in_spot': corner_in_spot, 'in_spot': in_spot}

    def map_action(self, action):
        # Function is to return acceleration and steering values from given action.
        # It must handle discrete and continuous cases

        assert self.action_space.contains(
            action), "%r (%s) invalid " % (action, type(action))

        if self.mode == 'discrete':
            accel = self.action_mapping['acceleration'][action]
            steer = self.action_mapping['steering'][action]
        else:
            accel = action[0]
            steer = action[1]

        return accel, steer

    def get_step_reward(self):
        step_reward = 0

        if self.parked:
            # reward for succeeding. This is the major task we wish to accomplish so this should
            # not be dominated by any other reward signal
            step_reward += PARKED_REWARD

        '''
        if self.first_spot_entry:
            # Reward for first time it gets a corner in the spot
            step_reward += 0.5 * PARKED_REWARD
            self.ep_message.append('Corner in spot')
        '''

        # Small reward for each corner the car gets into the spot
        for f in self.first_cis:
            step_reward += f * PARKED_REWARD / len(self.first_cis)

        # Reward getting close to the parking spot
        new_dist = self.get_dist_to_spot()

        step_reward += 3 * (self.old_dist - new_dist) / self.longest_dist * PARKED_REWARD
        self.old_dist = copy.copy(new_dist)

        # Reward (penalty) for keeping wheels straight (turning)
        # step_reward -= np.abs(self.car.steering) + 0.05

        # smooth_reward = 0.05
        # Reward for smooth actions
        # for i in range(len(self.last_action)):
        #    step_reward += 0
        #    step_reward += smooth_reward * np.exp(-20 * np.abs((action[i]-self.last_action[i])))

        # self.last_action = list([action])

        # Time penalty to encourage algorithm to finish
        # step_reward -= 0.5

        return step_reward

    def get_dist_to_spot(self):
        return np.sqrt((self.spot_x_loc - self.car.location[0])**2 +
                       (self.spot_y_loc - self.car.location[1])**2)
