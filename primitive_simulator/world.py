# world.py
# A continuous-state environment where all features are rectangular (does not provide support for rotated features).
# James D. Wood
# NeuroCar - Rogues' Gallery Neuro Team
# 8 October 2019

class World:
    """
    A World is a representation of the physical space in which this simulation takes place.
    A World keeps track of what features and objects are placed within it.
    A World is rectangular.
    """

    def __init__(self, width: float, height: float):
        """
        Creates a new World with given parameters width and height.
        """
        self.width = width
        self.height = height
        self.obstacles = []
        self.vehicle = None

    def add_obstacle(self, position: tuple, width: float, height: float):
        """
        Creates a new Obstacle and places it in the World at the given position.
        Obstacles are allowed to collide with other Obstacles.
        Obstacles cannot be created outside of the World bounds.
        """
        if position[0] - width / 2 >= 0 and position[0] + width / 2 <= self.width and position[1] - height / 2 >= 0 and position[1] + height / 2 <= self.height:
            self.obstacles.append(Obstacle(position, width, height))
            return self.obstacles[len(self.obstacles) - 1]
        else:
            print("WARNING: Obstacle unsuccessfully added! Would be outside of World bounds.")
            return None

    def initialize_vehicle(self, position: tuple, width: float, height: float):
        """
        Initializes the Vehicle of this World at the given position with given parameters width and height.
        Vehicles are not allowed to collide with Obstacles.
        Vehicles cannot be created outside of the World bounds.
        """
        if position[0] - width / 2 >= 0 and position[0] + width / 2 <= self.width and position[1] - height / 2 >= 0 and position[1] + height / 2 <= self.height:
            vehicle = Vehicle(position, width, height)
            if not self.check_collision(vehicle.get_corners()):
                self.vehicle = vehicle
                return self.vehicle
            else:
                print("WARNING: Vehicle unsuccessfully initialized! Would collide with Obstacle.")
                return None
        else:
            print("WARNING: Vehicle unsuccessfully initialized! Would be outside of World bounds.")
            return None

    def check_collision_position(self, position: tuple):
        """
        Checks if a given position is inside or on the border of one of the Obstacles within this World.
        Returns True if the position is inside or on the border of an Obstacle, or False if it is not.
        """
        for obstacle in self.obstacles:
            if obstacle.check_collision_position(position):
                return True
        return False

    def check_collision(self, positions: list):
        """
        Checks if any of the given positions are inside or on the border of one of the Obstacles within this World.
        Returns True if any position is inside or on the border of an Obstacle, or False if none are.
        """
        for obstacle in self.obstacles:
            if obstacle.check_collision(positions):
                return True
        return False

    def update(self, delta_t: float):
        """
        Will update the position of the contained Vehicle based upon its velocity.
        """
        if self.vehicle:
            self.vehicle.update(delta_t)

class Feature:
    """
    A Feature is any Object in the world which is rectangular and has a center, width, and height.
    """
    def __init__(self, center: tuple, width: float, height: float):
        """
        Creates a new Feature with given parameters center, width, and height.
        """
        self.center = center
        self.width = width
        self.height = height

    def get_upper_left_corner(self):
        """
        Gets the position of the upper-left corner of this Feature (north-west direction).
        """
        return () + (self.center[0] - self.width / 2, self.center[1] - self.height / 2)

    def get_upper_right_corner(self):
        """
        Gets the position of the upper-right corner of this Feature (north-east direction).
        """
        return () + (self.center[0] + self.width / 2, self.center[1] - self.height / 2)

    def get_lower_left_corner(self):
        """
        Gets the position of the lower-left corner of this Feature (south-west direction).
        """
        return () + (self.center[0] - self.width / 2, self.center[1] + self.height / 2)

    def get_lower_right_corner(self):
        """
        Gets the position of the lower-right corner of this Feature (south-east direction).
        """
        return () + (self.center[0] + self.width / 2, self.center[1] + self.height / 2)

    def get_corners(self):
        """
        Gets the corners of this Feature in this order: [upper_left, upper_right, lower_left, lower_right].
        """
        return [self.get_upper_left_corner, self.get_upper_right_corner, self.get_lower_left_corner, self.get_lower_right_corner]

class Obstacle(Feature):
    """
    An Obstacle is a feature in the World which can be collided with.
    Obstacles have a center position, a width, and a height. 
    As always, features are rectangular.
    """

    def __init__(self, center: tuple, width: float, height: float):
        """
        Creates a new Obstacle with given parameters center, width, and height.
        """
        self.center = center
        self.width = width
        self.height = height

    def check_collision_position(self, position: tuple):
        """
        Checks if the provided postition is inside or on the border of this Obstacle.
        Returns True if the position is inside or on the border, or False if it is not.
        """
        lower_left_corner = self.get_lower_left_corner()
        upper_right_corner = self.get_upper_right_corner()
        upper_left_corner = self.get_upper_left_corner()
        x_extent = () + (lower_left_corner[0], upper_right_corner[0])
        y_extent = () + (upper_left_corner[1], lower_left_corner[1])
        if position[0] >= x_extent[0] and position[0] <= x_extent[1] and position[1] >= y_extent[0] and position[1] <= y_extent[1]:
            return True
        return False

    def check_collision(self, positions: list):
        """
        Checks if any of the given positions are inside or on the border of this Obstacle.
        Returns True if any of the positions are inside or on the border, or False if none are.
        """
        for position in positions:
            if self.check_collision_position(position):
                return True
        return False

class Vehicle(Feature):
    """
    A Vehicle is represented as a rectangular shape and maintains a position and velocity.
    A Vehicle has vision in a certain range around it, and this vision is divided
    into several discrete "chunks" which are an analog to pixels.
    """

    def __init__(self, position: tuple, width: tuple, height: tuple):
        """
        Creates a new Vehicle with given parameters position, width, and height.
        The Vehicle also starts with a velocity of (0.0, 0.0).
        """
        self.center = position
        self.width = width
        self.height = height
        self.velocity = () + (0.0, 0.0)
        self.acceleration = () + (0.0, 0.0)

    def apply_acceleration(self, acceleration: tuple):
        """
        Applies an acceleration to the Vehicle where acceleration is the instantaneous change in velocity.
        """
        self.acceleration = acceleration

    def on_collision(self):
        """
        Function is called when this Vehicle collides with an Obstacle.
        """
        self.velocity = () + (0.0, 0.0)

    def update(self, delta_t: float):
        """
        Updates the velocity of the Vehicle based upon its current accleration.
        Updates the position of the Vehicle based upon its current velocity.
        Any acceleration should be applied prior to updating.
        """
        self.velocity = () + (self.velocity[0] + self.acceleration[0] * delta_t, self.velocity[1] + self.acceleration[1] * delta_t)
        self.center = () + (self.center[0] + self.velocity[0] * delta_t, self.center[1] + self.velocity[1]* delta_t)
        self.acceleration = () + (0.0, 0.0)
