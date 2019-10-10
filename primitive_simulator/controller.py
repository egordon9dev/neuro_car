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

class Controller:
    """
    Manages the interaction between a World and View.
    Controls the flow of the simulator. 
    """

    def __init__(self, dimensions: tuple, title: str):
        """
        Initializes the simulator with a given size and name.
        """
        self.world = World(dimensions[0], dimensions[1])
        self.view = View(self.world, title)

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

if __name__ == '__main__':
    controller = Controller((1280, 720), "Primitive Simulator")
    should_close = False
    while not should_close:
        try:
            if controller.view.check_mouse():
                should_close = True
        except GraphicsError:
            print("WARNING: Window was closed by user.")
            should_close = True
