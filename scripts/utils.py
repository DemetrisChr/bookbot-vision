import numpy as np
import cv2 as cv


class Line:
    """
    Class to represent a line with equation rho = xcos theta + ysin theta.
    rho is the perpendicular distance from origin to the line and theta is the
    angle formed by this perpedicular line and the x-axis measured counter-clockwise
    To represent it in the form ax + by + c =  we take a = cos theta, b = sin theta, c = -rho
    """
    def __init__(self, theta, rho):
        self.theta = theta
        self.rho = rho
        self.a = np.cos(theta)
        self.b = np.sin(theta)
        self.c = -rho

    def distanceFromPoint(self, point):
        """
        Calculates the distance of a given point (x, y) from the line using the formula
        distance = |ax + by + c| / (a^2 + b^2)
        """
        x0, y0 = point
        return np.abs(self.a * x0 + self.b * y0 + self.c) / np.linalg.norm((self.a, self.b))

    def plotOnImage(self, img, colour=(255, 0, 0), thickness=5):
        """
        Plots the line on the given image.
        """
        x0 = self.a * self.rho
        y0 = self.b * self.rho
        pt1 = (int(x0 + 1000 * (-self.b)), int(y0 + 1000 * (self.a)))
        pt2 = (int(x0 - 1000 * (-self.b)), int(y0 - 1000 * (self.a)))
        cv.line(img, pt1, pt2, colour, thickness, cv.LINE_AA)

    def calculateXgivenY(self, y):
        """
        Calculates the x-coordinate of a point on the line with the given y-coordinate.
        """
        return -(self.b * y + self.c) / self.a

    def calculateYgivenX(self, x):
        """
        Calculates the y-coordinate of a point on the line with the given x-coordinate.
        """
        return -(self.a * x + self.c) / self.b


class Rectangle:
    """
    Class to represent a rectangle with sides parallel to the coordinate axes.
    """
    def __init__(self, x, y, w, h):
        self.x = x  # x-coordinate of top left point
        self.y = y  # y-coordinate of top left point
        self.w = w  # width
        self.h = h  # height

    def unpack(self):
        """
        Returns the attributes of a rectange as a tuple
        """
        return (self.x, self.y, self.w, self.h)

    def isInnerRectangle(self, rectangles):
        """
        Returns whether this rectangle is within any of the rectangles passed
        as arguments
        """
        for rectangle in rectangles:
            (x, y, w, h) = rectangle.unpack()
            if x < self.x and (self.x + self.w) < (x + w) and \
                    y < self.y and (self.y + self.h) < (y + h):
                return True
        return False

    def plotOnImage(self, img, colour=(0, 255, 0), thickness=5):
        """
        Plots the rectangle on the given image
        """
        cv.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), colour, thickness)


def displayImage(img, rectangles=None, lines=None):
    """
    Displays image with lines and rectangles (if there are any)
    """
    cv.namedWindow("Display window", cv.WINDOW_AUTOSIZE)
    img_display = img.copy()

    if rectangles:
        for rectangle in rectangles:
            rectangle.plotOnImage(img_display)

    if lines:
        for line in lines:
            line.plotOnImage(img_display)

    M, N = img_display.shape[0], img_display.shape[1]
    img = cv.resize(img_display, (int(N / 4), int(M / 4)))
    img_display = cv.pyrDown(img_display, img_display)
    cv.imshow('Display window', img_display)
    cv.waitKey(0)
    return img_display


def removeInnerRectangles(rectangles):
    """
    Removes all inner rectangles from a list of rectangles. A rectangle is
    considered inner if it is within any other rectangle from the list.
    """
    return list(filter(lambda rec: not rec.isInnerRectangle(rectangles), rectangles))


def movingAverage(values, window_size):
    """
    Calculates the moving average of a given list of values
    """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(values, window, 'same')


def findClosestLineToPoint(point, lines):
    """
    Finds the line closest to the point from a given list of lines
    """
    if len(lines) == 0:
        return None
    min_distance = lines[0].distanceFromPoint(point)
    closest_line = lines[0]
    for line in lines[1:]:
        distance = line.distanceFromPoint(point)
        if distance < min_distance:
            min_distance = distance
            closest_line = line
    return closest_line
