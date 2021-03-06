# main.py
# A controller which manages the primitive simulation components.
# James D. Wood
# NeuroCar - Rogues' Gallery Neuro Team
# 10 October 2019

from world import World
from world import Obstacle
from world import Vehicle
from view import View
from graphics import GraphicsError

class Agent:
    """
    An Agent dictates commands to a Vehicle based upon built-in rules.
    """

    def __init__(self, vehicle: Vehicle, world: World):
        """
        Creates a new basic Agent with a given Vehicle to manage.
        """
        self.vehicle = vehicle
        self.world = World

    def sense(self):
        """
        Agents that can receive sensory data will Sense first
        in order to make decisions.
        Note: a basic Agent does not receive sensory data.
        """
        pass

    def act(self):
        """
        Agents perform internal calculations and modify the Vehicle's
        behavior within this method.
        Note: a basic Agent simply accelerates the Vehicle at a fixed rate.
        """
        self.vehicle.apply_acceleration((5.0, 0.0))

    def update(self):
        """
        Update performs the sense and act operations in sequence.
        Intended to be called each cycle.
        """
        self.sense()
        self.act()

class LapAgent(Agent):
    """
    An Agent which loops the track.
    """

    def __init__(self, vehicle: Vehicle, world: World):
        """
        Creates a new LapAgent with a given Vehicle to manager and the World.
        """
        super(LapAgent, self).__init__(vehicle, world)
        self.world = world

    def act(self):
        """
        This act stage will figure out the necessary velocity and apply it.
        Kinda cheating but I need it for testing.
        """
        center = self.vehicle.center
        if center[0] <= 250 and center[1] >= 250:
            self.vehicle.velocity = () + (0.0, -100.0)
        elif center[0] <= 250 and center[1] < 250:
            self.vehicle.velocity = () + (100.0, 0.0)
        elif center[0] >= 1030 and center[1] < 250:
            self.vehicle.velocity = () + (0.0, 100.0)
        elif center[0] <= 1030 and center[1] >= 470:
            self.vehicle.velocity = () + (-100.0, 0.0)

class Controller:
    """
    Manages the interaction between a World and View.
    Controls the flow of the simulator. 
    """

    def __init__(self, dimensions: tuple, title: str, delta_t: float):
        """
        Initializes the simulator with a given size and name.
        delta_t is the timestep of the simulation.
        """
        self.world = World(dimensions[0], dimensions[1])
        self.view = View(self.world, title)
        self.delta_t = delta_t
        self.agent = None

    def add_obstacle(self, position: tuple, width: float, height: float, color: str):
        """
        Adds an Obstacle to the simulation at the given location
        with given dimensions and a given color.
        """
        obstacle = self.world.add_obstacle(position, width, height)
        if obstacle:
            self.view.add_feature(obstacle.get_upper_left_corner(), obstacle.get_lower_right_corner(), color)

    def initialize_vehicle(self, position: tuple, width: float, height: float, color: str):
        """
        Initializes the Vehicle in the simulation at a given location
        with given dimensions and color.
        """
        vehicle = self.world.initialize_vehicle(position, width, height)
        if vehicle:
            self.view.set_vehicle(vehicle.get_upper_left_corner(), vehicle.get_lower_right_corner(), color)

    def set_agent(self, agent: Agent):
        """
        Sets the Agent that will control the Vehicle.
        """
        self.agent = agent

    def update(self):
        """
        Updates the World and View components of this simulation. 
        """
        if self.agent:
            self.agent.update()
        self.world.update(self.delta_t)
        self.view.update(self.delta_t)

if __name__ == '__main__':
    controller = Controller((1280, 720), "Primitive Simulator", 0.001)
    controller.initialize_vehicle((100, 100), 40, 40, "red")
    controller.set_agent(LapAgent(controller.world.vehicle, controller.world))
    controller.add_obstacle((25, 360), 50, 720, "blue")
    controller.add_obstacle((1255, 360), 50, 720, "blue")
    controller.add_obstacle((640, 25), 1180, 50, "blue")
    controller.add_obstacle((640, 695), 1180, 50, "blue")
    controller.add_obstacle((640, 360), 450, 300, "blue")
    #print(controller.world.convert_area_to_pixel((640, 360), 1280, 720))
    #print(controller.world.convert_area_to_pixel_array((10, 10), 37, 65, 20))
    #print(controller.world.get_point_distances_from_obstacles((100, 100)))
    #controller.world.get_point_distances_from_obstacles((100, 100))
    #controller.world.get_vehicle_sensory_data(5, 5, 5)

    should_close = False
    while not should_close:
        try:
            if controller.view.check_mouse():
                should_close = True
            else:
                controller.update()
                print(controller.world.get_vehicle_sensory_data(50, 5, 5))
        except GraphicsError as ge:
            print("WARNING: " + ge.__str__())
            should_close = True
        except Exception as e:
            print(e)
            should_close = True
    #controller.view.capture_png("test")
    controller.view.window.close()
