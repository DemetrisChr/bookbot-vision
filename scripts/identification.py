import numpy as np
import cv2 as cv
import time
import pytesseract
from utils import Rectangle, displayImage, movingAverage
from scipy.signal import find_peaks, peak_widths
from book_match import BookMatch, label_codes_original
from threading import Thread


class Book:
    def __init__(self, label_rectangle, label_img):
        self.label_rectangle = label_rectangle
        self.label_img = label_img
        self.label_img_preprocessed = None
        self.label_ocr_text = None
        self.matched_lcc_code = None
        self.match_cost = None

    def parseBookLabel(self):
        """
        Does some preprocessing to the image of the label and uses OCR to read the text.
        """
        # Binarise the image
        img = rowLuminosityBinarisation(self.label_img, num_intervals=1, threshold_coef=0.6).astype('uint8')

        # Resize image
        scale_percent = 200  # Percentage of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

        # Add white border around the image
        img = cv.copyMakeBorder(img, top=50, bottom=50, left=50, right=50, borderType=cv.BORDER_CONSTANT, value=255)

        # Apply Gaussian blurring
        img = cv.GaussianBlur(img, (5, 5), 2)

        # Dilate the image
        kernel = np.ones((5, 5), np.uint8)
        img = cv.dilate(img, kernel, iterations=1)

        self.label_img_preprocessed = img
        self.label_ocr_text = pytesseract.image_to_string(self.label_img_preprocessed)  # , config='bazaar --oem 0'))

    def findMatch(self, all_labels):
        """
        Finds the closest match to the label, using levenshtein distance
        """
        if len(self.label_ocr_text) > 0:
            bm = BookMatch(all_labels=all_labels)
            self.matched_lcc_code, self.match_cost = bm.closest_label_match(self.label_ocr_text)

    def __str__(self):
        if self.matched_lcc_code:
            return self.label_ocr_text + \
                '\nBest match:    ' + self.matched_lcc_code + \
                '\nEdit distance: ' + str(self.match_cost)
        elif self.label_ocr_text == '':
            return 'EMPTY LABEL'
        else: 
            return 'NO MATCH FOUND' + \
                '\nEdit distance:    ' + str(self.match_cost) + \
                '\n' + self.label_ocr_text


class BooksImage:
    def __init__(self, filename=None, camera_idx=0):
        self.img_bgr = None
        if filename is None:
            # Take picture from camera
            print('Taking picture from camera...')
            video_capture = cv.VideoCapture(index=camera_idx)
            if not video_capture.isOpened():
                raise Exception('Could not open video device')
            ret, self.img_bgr = video_capture.read()  # Read picture
            video_capture.release()  # Close device
        else:
            if isinstance(filename, str):
                self.img_bgr = cv.imread(filename)
            else:
                self.img_bgr = filename
        self.img_rgb = cv.cvtColor(self.img_bgr, cv.COLOR_BGR2RGB)
        self.img_gray = cv.cvtColor(self.img_bgr, cv.COLOR_BGR2GRAY)

        # For label detection
        self.img_binary = None
        self.img_eroded = None

        self.books = []

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
        # cv.imwrite('test.png', self.img_binary)

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
        return self.img_eroded

    def findRowBounds(self):
        """
        Finds the rows where the labels could be located
        """
        # TODO Ignore internal row bounds
        count_row = np.sum(self.img_eroded, axis=1) / 255  # Number of white pixels in each row
        moving_average_row = movingAverage(count_row, 100)  # Moving average of white pixels for each row

        min_height = int(self.N / 10)  # Minimum number of white pixels for row to be considered for local max
        peaks, _ = find_peaks(moving_average_row, height=min_height)  # Finds the local maxima
        _, _, top, bottom = peak_widths(moving_average_row, peaks, rel_height=0.5)
        self.row_bounds = list(zip(top.astype('int'), bottom.astype('int')))
        self.row_bounds = list(filter(lambda r: r[1] - r[0] > self.M / 20, self.row_bounds))
        print(self.row_bounds)

    def findLabelsWithinBounds(self, top, bottom, contour_approx_strength=0.05):
        """
        Finds labels that are within the specified bounds
        """
        min_contour_area = self.M * self.N / 100
        max_contour_area = self.M * self.N / 6
        _, contours, hierarchy = cv.findContours(self.img_eroded[top:bottom, :], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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
            self.findLabelsWithinBounds(top, bottom, contour_approx_strength=contour_approx_strength)

        # self.label_rectangles = removeInnerRectangles(self.label_rectangles)

    def parseLabels(self, all_labels):
        """
        Uses Optical Character Recognition (OCR) to parse the text from the
        labels.
        """
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # For Windows only

        threads = []
        books = []
        for rectangle in self.label_rectangles:
            (x, y, w, h) = rectangle.unpack()
            img_slice = self.img_gray[y:y + h, x:x + w].copy()
            book = Book(rectangle, img_slice)
            thread = Thread(target=book.parseBookLabel)
            thread.start()
            threads.append(thread)
            books.append(book)

        # counter = 1
        for i in range(len(books)):
            threads[i].join()
            books[i].findMatch(all_labels)
            self.books.append(books[i])
            # cv.imwrite('label' + str(counter) + '.png', books[i].label_img_preprocessed)
            # counter += 1

    def preprocessAndReadLabels(self, all_labels):
        self.generateBinaryImage(num_intervals=20, threshold_coef=0.85)
        self.erodeBinaryImage()
        self.findLabels()
        self.parseLabels(all_labels)
        # displayImage(self.img_binary, rectangles=self.label_rectangles)
        # displayImage(self.img_eroded, rectangles=self.label_rectangles)
        # displayImage(self.img_bgr, rectangles=self.label_rectangles)


def rowLuminosityBinarisation(img, num_intervals, threshold_coef):
    """
    Binarises an image by using binary thresholding. The threshold is
    calculated as the product of threshold_coef and the maximum row
    luminosity. Luminosity of a row is the mean intensity of its pixels.
    """
    M, N = img.shape
    row_luminosities = []
    for row_idx in range(M):
        luminosity = np.sum(img[row_idx, :]) / N
        row_luminosities.append(luminosity)
    max_row_luminosity = np.max(row_luminosities)
    threshold = threshold_coef * max_row_luminosity
    return 255 * (img > threshold)


def findBook(booksimg, target_lcc_code):
    """
    Finds the book with the given LCC code in the image and displays its boundaries
    """
    target_book = None
    min_cost = 100
    for book in booksimg.books:
        # print('tagrget=' + target_lcc_code)
        # print('match=  ' + str(book.matched_lcc_code))
        if book.matched_lcc_code == target_lcc_code and book.match_cost < min_cost:
            min_cost = book.match_cost
            target_book = book
    if target_book is not None:
        print(str(target_book.matched_lcc_code) + ' has been found!')
        print('    Book label location: ' + str(target_book.label_rectangle.unpack()))
        return target_book.label_rectangle
        # img_display = displayImage(booksimg.img_bgr, rectangles=[target_book.label_rectangle])
        # cv.imwrite(target_title + '.png', img_display)
    else:
        print(target_lcc_code + ' could not be found :(')
        return None


def main():
    start_time = time.time()
    booksimg = BooksImage('../pictures/webcam7.jpg')
    booksimg.preprocessAndReadLabels(label_codes_original)

    for book in booksimg.books:
        print('=============')
        print(book)
    print('==========================')
    print('TOTAL RUNTIME: %s seconds' % (time.time() - start_time))
    print('==========================')

    displayImage(booksimg.img_binary, rectangles=booksimg.label_rectangles)
    displayImage(booksimg.img_eroded, rectangles=booksimg.label_rectangles)
    displayImage(booksimg.img_bgr, rectangles=booksimg.label_rectangles)

    print('\n==========================\n')
    for book_code in label_codes_original:
        findBook(booksimg, book_code)
        print('==========================')
    return booksimg


if __name__ == '__main__':
    main()
