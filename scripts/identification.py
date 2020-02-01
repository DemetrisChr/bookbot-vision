import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time
import pytesseract
from scipy.signal import find_peaks, peak_widths


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
        self.img_eroded = None
        self.label_codes = []
        self.label_rectangles = []
        self.M, self.N = self.img_gray.shape  # M - rows, N - columns
        self.row_bounds = None

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
            start = int(interval_idx * interval_width)
            end = int((interval_idx + 1) * interval_width)
            self.img_binary[:, start:end] = rowLuminosityBinarisation(self.img_binary[:, start:end], num_intervals, threshold_coef)

    def erodeBinaryImage(self, kernel_shape=(5, 5), iterations=1):
        """
        Applies erosion to the binary image in order to remove small white
        regions which can also be joined to the labels, which can be
        problematic when detecting the locations of the labels. The labels can
        still be detected since we expect them to be large white regions.
        """
        kernel = np.ones(kernel_shape, np.uint8)
        self.img_binary = self.img_binary.copy()
        self.img_eroded = cv.erode(self.img_binary, kernel, iterations)

    def findRowBounds(self):
        """
        Finds the rows where the labels could be located
        """
        count_row = np.sum(self.img_binary, axis=1) / 255  # Number of white pixels in each row
        moving_average_row = movingAverage(count_row, 200)  # Moving average of white pixels for each row

        min_height = int(self.N / 3)  # Minimum number of white pixels for row to be considered for local max
        peaks, _ = find_peaks(moving_average_row, height=min_height)  # Finds the local maxima
        _, _, top, bottom = peak_widths(moving_average_row, peaks, rel_height=0.95)
        self.row_bounds = list(zip(top.astype('int'), bottom.astype('int')))
        print(self.row_bounds)

    def findLabelsWithinBounds(self, top, bottom, contour_approx_strength=0.05):
        """
        Finds labels that are within the specified bounds
        """
        min_contour_area = self.M * self.N / 600
        max_contour_area = self.M * self.N / 25
        contours, hierarchy = cv.findContours(self.img_eroded[top:bottom, :], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        largest_contours = list(filter(lambda c: cv.contourArea(c) >= min_contour_area and cv.contourArea(c) <= max_contour_area, contours))

        approximated_contours = []
        for contour in largest_contours:
            epsilon = contour_approx_strength * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            approximated_contours.append(approx)

        for contour in approximated_contours:
            x, y, w, h = cv.boundingRect(contour)
            y += top
            self.label_rectangles.append(Rectangle(x, y, w, h))

    def findLabels(self, contour_approx_strength=0.05):
        """
        Locates the labels using the binary image
        """
        self.findRowBounds()
        for top, bottom in self.row_bounds:
            self.findLabelsWithinBounds(top, bottom)

        # self.label_rectangles = removeInnerRectangles(self.label_rectangles)

    def parseLabels(self):
        """
        Uses Optical Character Recognition (OCR) to parse the text from the
        labels.
        """
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        counter = 1
        for rectangle in self.label_rectangles:
            (x, y, w, h) = rectangle.unpack()
            img_slice = self.img_binary[y:y + h, x:x + w]
            cv.imwrite('label' + str(counter) + '.png', self.img_gray[y:y + h, x: x + w])

            # Resize image
            scale_percent = 200  # Percentage of original size
            width = int(img_slice.shape[1] * scale_percent / 100)
            height = int(img_slice.shape[0] * scale_percent / 100)
            dim = (width, height)
            img_slice = cv.resize(img_slice, dim, interpolation=cv.INTER_AREA)

            # Add white border around the image
            img_slice = cv.copyMakeBorder(img_slice, top=100, bottom=100, left=100, right=100, borderType=cv.BORDER_CONSTANT, value=255)

            # Dilate the image
            kernel = np.ones((5, 5), np.uint8)
            img_slice = cv.dilate(img_slice, kernel, iterations=1)

            self.label_codes.append(pytesseract.image_to_string(img_slice, config='config'))
            counter += 1


def rowLuminosityBinarisation(img, num_intervals, threshold_coef):
    """
    Binarises an image by using binary thresholding. The threshold is
    calculated as the product of threshold_coef and the maximum row
    luminosity. Luminosity of a row is the mean intensity of its pixels
    """
    M, N = img.shape
    row_luminosities = []
    for row_idx in range(M):
        luminosity = np.sum(img[row_idx, :]) / N
        row_luminosities.append(luminosity)
    max_row_luminosity = np.max(row_luminosities)
    threshold = threshold_coef * max_row_luminosity
    return 255 * (img > threshold)


def displayImage(img, cmap='gray', rectangles=None):
    """
    Displays an image within a matplotlib figure alongside any rectangles
    passed to this function.
    """
    plt.figure(figsize=(16, 12))
    if rectangles:
        for rectangle in rectangles:
            (x, y, w, h) = rectangle.unpack()
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    plt.imshow(img, cmap)
    plt.show()


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


if __name__ == '__main__':
    start_time = time.time()
    books = BooksImage('../notebooks/pictures/books5.jpg')
    books.generateBinaryImage(num_intervals=20)
    books.erodeBinaryImage()
    books.findLabels()
    books.parseLabels()
    for label in books.label_codes:
        print(label)
        print('-----')
    print("--- %s seconds ---" % (time.time() - start_time))
    displayImage(books.img_binary, rectangles=books.label_rectangles)
    displayImage(books.img_bgr, rectangles=books.label_rectangles)
