import numpy as np
import math
import time
from Box2D import b2FixtureDef, b2PolygonShape

# Top-down car dynamics simulation.
#
# Some ideas are taken from this great tutorial
# http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
# This simulation is a bit more detailed, with wheels rotation.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

SIZE = 0.02

# Calibration factors
ACCELERATOR_SCALING_FACTOR = 2.0
STEERING_SCALING_FACTOR = 0.05
CAR_FRICTION = 0.01
STEERING_ANGLE_LIMIT = math.pi / 6
SKEW_FACTOR = 2

Body_Poly = [
    (-60, +110), (-60, -110),
    (-50, -120), (+50, -120),
    (+60, -110), (+60, +110),
    (+40, +130), (-40, +130)]
Left_Mirror_Poly = [
    (-60, +45), (-70, +45),
    (-75, +35), (-60, +35)]
Right_Mirror_Poly = [
    (+60, +45), (+70, +45),
    (+75, +35), (+60, +35)]
Rear_Window_Poly = [
    (-40, -115), (+40, -115),
    (+45, -85), (-45, -85)]
Windshield_Poly = [
    (-50, +70), (-30, +80),
    (+30, +80), (+50, +70),
    (+45, +20), (-45, +20)]
Left_Window_Poly = [
    (-55, +60), (-55, -70),
    (-45, -75), (-45, +5)]
Right_Window_Poly = [
    (+55, +60), (+55, -70),
    (+45, -75), (+45, +5)]
Left_Taillight_Poly = [
    (-40, -115), (-60, -110),
    (-60, -85), (-55, -85)]
Right_Taillight_Poly = [
    (+40, -115), (+60, -110),
    (+60, -85), (+55, -85)]
Left_Headlight_Poly = [
    (-60, +110), (-40, +130),
    (-35, +130), (-60, +100)]
Right_Headlight_Poly = [
    (+60, +110), (+40, +130),
    (+35, +130), (+60, +100)]


class Car:
    def __init__(self, world, init_angle, init_x, init_y, max_speed):
        self.world = world
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                b2FixtureDef(shape=b2PolygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in Body_Poly]), density=1.0),
                b2FixtureDef(shape=b2PolygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in Left_Mirror_Poly]), density=1.0),
                b2FixtureDef(shape=b2PolygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in Right_Mirror_Poly]), density=1.0),
                b2FixtureDef(shape=b2PolygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in Rear_Window_Poly]), density=1.0),
                b2FixtureDef(shape=b2PolygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in Windshield_Poly]), density=1.0),
                b2FixtureDef(shape=b2PolygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in Left_Window_Poly]), density=1.0),
                b2FixtureDef(shape=b2PolygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in Right_Window_Poly]), density=1.0),
                b2FixtureDef(shape=b2PolygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in Left_Taillight_Poly]), density=1.0),
                b2FixtureDef(shape=b2PolygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in Right_Taillight_Poly]), density=1.0),
                b2FixtureDef(shape=b2PolygonShape(
                    vertices=[(x * SIZE, y * SIZE) for x, y in Left_Headlight_Poly]), density=1.0),
                b2FixtureDef(shape=b2PolygonShape(vertices=[(x * SIZE, y * SIZE) for x, y in Right_Headlight_Poly]), density=1.0)])

        self.hull.angle =  init_angle
        self.hull.userData = self.hull
        self.hull.colors = {
            'Body': (26 / 255, 198 / 255, 255 / 255),
            'Window': (38 / 255, 38 / 255, 38 / 255),
            'Taillight': (230 / 255, 46 / 255, 0 / 255),
            'Headlight': (255 / 255, 255 / 255, 255 / 255)}

        self.drawlist = [self.hull]
        self.speed = np.random.uniform(0, max_speed, 1)[0]
        #self.speed = 0

        self.max_speed = max_speed
        # self.steering=init_angle
        self.steering = 0
        self.steering_limit = STEERING_ANGLE_LIMIT

        self.corners = [(x * SIZE, y * SIZE) for x, y in Body_Poly]
        self.location = [init_x, init_y]

    def _reset(self, initial_x, initial_y, initial_angle, initial_speed):
        self.hull.position[0] = initial_x
        self.hull.position[1] = initial_y
        self.location = [self.hull.position[0], self.hull.position[1]]
        self.steering = 0
        self.hull.angle = initial_angle
        self.speed = np.clip(initial_speed,0,self.max_speed)

    def accelerator(self, acceleration, dt):
        if acceleration < 0:
            acc = acceleration ** (2 * SKEW_FACTOR + 1)
        else:
            acc = acceleration ** (1 / (2 * SKEW_FACTOR + 1))
        #acc = (acceleration / np.abs(acceleration)) * acceleration ** 2
        self.speed += (ACCELERATOR_SCALING_FACTOR *
                       acc - CAR_FRICTION) * (dt**2)
        self.speed = np.minimum(np.maximum(self.speed, 0), self.max_speed)


    def steering_rate(self, steering_rate):
    	# This line has the steering revert to center
        # self.steering = 0.9 * self.steering + steering_rate * STEERING_SCALING_FACTOR
        steering_rate = steering_rate ** 9
        self.steering += steering_rate * STEERING_SCALING_FACTOR
        self.steering = np.clip(
            self.steering, -self.steering_limit, self.steering_limit)
        

    def step(self, dt):

        o1_start = time.time()
        tan_ang = math.tan(self.steering)
        add_thing = dt * self.speed * tan_ang
        o1_time = time.time() - o1_start

        o2_start = time.time()
        self.hull.angle += add_thing
        o2_time = time.time() - o2_start
        
        #print('Hull angle:', self.hull.angle / math.pi, 'Mod 2 Pi:', (self.hull.angle % (math.pi)) / math.pi)
        if self.hull.angle > math.pi:
            self.hull.angle -= 2 * math.pi
        if self.hull.angle < -math.pi:
            self.hull.angle += 2 * math.pi

        o3_start = time.time()
        self.hull.position[0] += float(dt * self.speed * math.cos(
            self.hull.angle + math.pi / 2))
        self.hull.position[1] += float(dt * self.speed * math.sin(
            self.hull.angle + math.pi / 2))
        self.location = [self.hull.position[0], self.hull.position[1]]
        o3_time = time.time() - o3_start

        return o1_time, o2_time, o3_time

    def draw(self, viewer):
        for fixt in range(len(self.hull.fixtures)):
            f = self.hull.fixtures[fixt]
            trans = f.body.transform
            path = [trans * v for v in f.shape.vertices]
            if fixt <= 2:
                viewer.draw_polygon(path, color=self.hull.colors['Body'])
            elif fixt >= 3 and fixt <= 6:
                viewer.draw_polygon(path, color=self.hull.colors['Window'])
            elif fixt >= 7 and fixt <= 8:
                viewer.draw_polygon(path, color=self.hull.colors['Taillight'])
            elif fixt >= 9 and fixt <= 10:
                viewer.draw_polygon(path, color=self.hull.colors['Headlight'])

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
