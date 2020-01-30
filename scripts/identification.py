import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


class Rectangle:
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


class BooksImage:
    def __init__(self, filename):
        self.img_bgr = cv.imread(filename)
        self.img_rgb = cv.cvtColor(self.img_bgr, cv.COLOR_BGR2RGB)
        self.img_gray = cv.cvtColor(self.img_bgr, cv.COLOR_BGR2GRAY)
        self.img_binary = None
        self.label_codes = []
        self.label_rectangles = []
        self.M, self.N = self.img_gray.shape  # M - rows, N - columns

    def generateBinaryImage(self, num_intervals=20, threshold_coef=0.85):
        """
        Converts the grayscale image to a binary image using the algorithm
        described in:
        https://www.researchgate.net/publication/225367956_The_UJI_librarian_robot
        (page 11)
        """
        interval_width = self.N / num_intervals
        self.img_binary = self.img_gray.copy()

        for interval_idx in range(num_intervals):
            interval_luminosities = []
            for row_idx in range(self.M):
                start = int(interval_idx * interval_width)
                end = int((interval_idx + 1) * interval_width)
                luminosity = np.sum(self.img_gray[row_idx, start:end]) \
                    / interval_width
                interval_luminosities.append(luminosity)
            interval_max_luminosity = np.max(interval_luminosities)

            threshold = threshold_coef * interval_max_luminosity
            self.img_binary[:, start:end] = 255 * \
                (self.img_binary[:, start:end] > threshold)

    def erodeBinaryImage(self, kernel_shape=(10, 10), iterations=1):
        """
        Applies erosion to the binary image in order to remove small white
        regions which can also be joined to the labels, which can be
        problematic when detecting the locations of the labels. The labels can
        still be detected since we expect them to be large white regions.
        """
        kernel = np.ones(kernel_shape, np.uint8)
        self.img_binary = cv.erode(self.img_binary, kernel, iterations)

    def findLabels(self, contour_approx_strength=0.05):
        """
        Locates the labels using the binary image
        """
        contours, hierarchy = cv.findContours(self.img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        largest_contours = list(filter(lambda c: cv.contourArea(c) >= 20000, contours))

        approximated_contours = []
        for contour in largest_contours:
            epsilon = contour_approx_strength * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            approximated_contours.append(approx)

        for contour in approximated_contours:
            self.label_rectangles.append(Rectangle(*cv.boundingRect(contour)))

        self.label_rectangles = removeInnerRectangles(self.label_rectangles)

    def parseLabels():
        """
        Uses Optical Character Recognition (OCR) to parse the text from the
        labels.
        """
        # TODO
        return


def displayImage(img, cmap='gray', rectangles=None):
    plt.figure(figsize=(16, 12))
    if rectangles:
        for rectangle in rectangles:
            (x, y, w, h) = rectangle.unpack()
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
    plt.imshow(img, cmap)
    plt.show()


def removeInnerRectangles(rectangles):
    return list(filter(lambda rec: not rec.isInnerRectangle(rectangles), rectangles))


if __name__ == '__main__':
    books = BooksImage('../notebooks/pictures/books1.jpg')
    books.generateBinaryImage()
    books.erodeBinaryImage()
    books.findLabels()
    displayImage(books.img_binary, rectangles=books.label_rectangles)
    displayImage(books.img_bgr, rectangles=books.label_rectangles)
