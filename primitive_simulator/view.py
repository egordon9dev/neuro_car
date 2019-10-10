# view.py
# A graphical representation of a World (based upon graphics.py).
# James D. Wood
# NeuroCar - Rogues' Gallery Neuro Team
# 8 October 2019

from graphics import Rectangle
from graphics import GraphWin
from graphics import Point
from world import World
from PIL import Image

class View:
    """
    Will handle all functions necessary to provide a graphical representation of a World.
    """

    def __init__(self, world: World, title: str):
        """
        Creates a new View with a given World to represent.
        """
        self.world = world
        self.window = GraphWin(title, world.width, world.height, autoflush=False)
        self.rectangles = []
        self.vehicle = None

    def update(self, delta_t: float):
        """
        Redraws this View (call upon update in coordinates).
        """
        if self.vehicle:
            self.vehicle.move(self.world.vehicle.velocity[0] * delta_t, self.world.vehicle.velocity[1] * delta_t)
        self.window.update()

    def add_feature(self, upper_left_corner: tuple, lower_right_corner: tuple, color: str):
        """
        Adds a feature to be drawn in this view.
        """
        self.rectangles.append(Rectangle(Point(upper_left_corner[0], upper_left_corner[1]), Point(lower_right_corner[0], lower_right_corner[1])))
        self.rectangles[len(self.rectangles) - 1].setFill(color)
        self.rectangles[len(self.rectangles) - 1].setOutline(color)
        self.rectangles[len(self.rectangles) - 1].draw(self.window)

    def set_vehicle(self, upper_left_corner: tuple, lower_right_corner: tuple, color: str):
        """
        Sets the location of the Vehicle.
        """
        self.vehicle = Rectangle(Point(upper_left_corner[0], upper_left_corner[1]), Point(lower_right_corner[0], lower_right_corner[1]))
        self.vehicle.setFill(color)
        self.vehicle.draw(self.window)

    def check_mouse(self):
        """
        Checks if a mouse button has been clicked during the previous frame.
        Returns the specific button if a button was clicked, or None if not button was pressed.
        """
        return self.window.checkMouse()

    def close_window(self):
        """
        Closes the window associated with this View.
        """
        self.window.close()

    def capture_png(self, name: str):
        """
        Records a postscript (similar to PDF) of the current graphical representation to the given file name.
        Converts this postscript to a png.
        """
        file_name = name + ".eps"
        self.window.postscript(file=file_name, colormode='color')
        img = Image.open(file_name)
        img.save(name + ".png", "png")