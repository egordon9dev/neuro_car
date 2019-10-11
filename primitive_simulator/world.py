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
        Obstacles are allowed to collide with other Obstacles (but it is discouraged due to potential accuracy errors it can introduce).
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
            updated_position = self.vehicle.get_updated_position_prior(delta_t)
            if self.check_collision(self.update_helper(updated_position)):
                self.vehicle.on_collision()
            else:
                self.vehicle.update(delta_t)

    def update_helper(self, updated_position: tuple):
        """
        Helper method designed to calculate the corners of the Vehicle
        based upon its updated position.
        """
        vehicle_width_half = self.vehicle.width / 2
        vehicle_height_half = self.vehicle.height / 2
        upper_left = () + (updated_position[0] - vehicle_width_half, updated_position[1] - vehicle_height_half)
        upper_right = () + (updated_position[0] + vehicle_width_half, updated_position[1] - vehicle_height_half)
        lower_left = () + (updated_position[0] - vehicle_width_half, updated_position[1] + vehicle_height_half)
        lower_right = () + (updated_position[0] + vehicle_width_half, updated_position[1] + vehicle_height_half)
        return [upper_left, upper_right, lower_left, lower_right]

    def convert_area_to_pixel(self, position: tuple, width: float, height: float):
        """
        For the purposes of getting "camera data," this simulator will perform
        a 2D analog of that by taking an area of the World and returning a value
        that reflects a combination of features (or lack thereof) in that area.
        """
        if not position[0] - width >= 0 and not position[0] + width <= self.width and not position[1] - height >= 0 and not position[1] + height <= self.height:
            return -1.0
        feature = Feature(position, width, height)
        percentages = []
        for obstacle in self.obstacles:
            percentages.append(obstacle.get_percentage_of_overlapping(feature))
        return min(sum(percentages), 1.0)

    def convert_area_to_pixel_array(self, starting_pixel_position: tuple, num_rows: int, num_cols: int, pixel_size: float):
        """
        Will convert a rectangular region of the World into a pixel array.
        """
        return_array = []
        for x in range(num_rows):
            for y in range(num_cols):
                current_position = () + (starting_pixel_position[0] + x * pixel_size, starting_pixel_position[1] + y * pixel_size)
                return_array.append(self.convert_area_to_pixel(current_position, pixel_size, pixel_size))
        return return_array

    def get_vehicle_sensory_data(self, pixel_size: float, radius: int):
        """
        Gets a pixel array starting at the Vehicle's position, utilizing pixel_size pixels
        radiating out in radius levels.
        """
        return self.convert_area_to_pixel_array(self.vehicle.center, radius, radius, pixel_size)

    def get_point_distances_from_obstacles(self, position: tuple):
        """
        Gets the distances to each obstacle in the World from a given point.
        """
        return_array = []
        for obstacle in self.obstacles:
            dx = max(abs(obstacle.center[0] - position[0]) - obstacle.width / 2, 0)
            dy = max(abs(obstacle.center[1] - position[1]) - obstacle.height / 2, 0)
            return_array.append((dx * dx + dy * dy) ** (.5))
        return return_array

    def get_vehicle_shortest_distance_to_obstacle(self):
        """
        Gets the distance between the Vehicle and the closest Obstacle.
        """
        return min(self.get_point_distances_from_obstacles(self.vehicle.center))

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
        return [self.get_upper_left_corner(), self.get_upper_right_corner(), self.get_lower_left_corner(), self.get_lower_right_corner()]

    def get_area(self):
        """
        Returns the area of this Feature.
        """
        return self.width * self.height

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

    def get_percentage_of_overlapping(self, feature: Feature):
        """
        This method is used for the purposes of converting the World space into a pixel space.
        This method finds the percentage of the passed in Feature overlapping with this Feature.
        Returns the percentage expressed as a float.
        """
        if not self.check_collision(feature.get_corners()):
            return 0.0


        upper_left = self.get_upper_left_corner()
        other_upper_left = feature.get_upper_left_corner()
        lower_right = self.get_lower_right_corner()
        other_lower_right = feature.get_lower_right_corner()
        
        intersect_upper_left = () + (max(upper_left[0], other_upper_left[0]), max(upper_left[1], other_upper_left[1]))
        intersect_lower_right = () + (min(lower_right[0], other_lower_right[0]), min(lower_right[1], other_lower_right[1]))
        
        intersecting_area = (intersect_lower_right[0] - intersect_upper_left[0]) * (intersect_lower_right[1] - intersect_upper_left[1])
        
        intersecting_area = max(intersecting_area, 0)

        return intersecting_area / feature.get_area()

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

    def get_updated_position_prior(self, delta_t: float):
        """
        For the purposes of checking for collisions, this method
        may be used to find the position of the Vehicle
        after it is updated, without actually updating its position.
        """
        velocity_temp = () + (self.velocity[0] + self.acceleration[0] * delta_t, self.velocity[1] + self.acceleration[1] * delta_t)
        return () + (self.center[0] + velocity_temp[0] * delta_t, self.center[1] + velocity_temp[1]* delta_t)
