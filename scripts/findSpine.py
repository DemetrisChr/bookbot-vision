from identification import Line, findClosestLineToPoint
import cv2 as cv
import numpy as np

def findBookBoundaries(image):
      """
      Finds all lines that could be book boundaries using Canny Edge Detection and Hough Line Transform
      """
      #
      img_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

      img_downsampled = None
      img_downsampled = cv.pyrDown(img_grey, img_downsampled)  # Downsample - scale factor 2
      img_canny_edge = cv.Canny(img_downsampled, 50, 50)
      hough_lines = cv.HoughLines(image=img_canny_edge, rho=1, theta=np.pi / 180, threshold=50)
      boundary_lines = []

      #TODO: Handle if no lines found (houghlines is NoneType)
      for hough_line in hough_lines:
          rho = hough_line[0][0]
          theta = hough_line[0][1]
          if abs(theta) < np.pi / 20 or abs(theta) > 19 * np.pi / 20:# (abs(theta) >= 0 and abs(theta) < np.pi / 20) or (abs(theta) >= (19 / 20) * np.pi and abs(theta) <= (21/20) * np.pi):
              # Keep only lines that are vertical or almost vertical
              # Rho is multiplied by 2 as the image used for detecting the lines is downsampled
              boundary_lines.append(Line(theta, 2 * rho))
      return boundary_lines


def findSpineBoundaries(image, label_rectangle):
      """
      Finds the left and right boundaries of the spine of this book
      """
      left_lines = []
      right_lines = []
      lines = findBookBoundaries(image)

      # Define point on the boundaries of the label
      midpoint_y = label_rectangle.y + label_rectangle.h / 2
      label_left_midpoint_x = label_rectangle.x
      label_right_midpoint_x = label_rectangle.x + label_rectangle.w

      # Split the line into lines lying on the right vs left of the label
      for line in lines:
          line_midpoint_x = line.calculateXgivenY(midpoint_y)
          if line_midpoint_x < label_left_midpoint_x:
              left_lines.append(line)
          elif line_midpoint_x > label_right_midpoint_x:
              right_lines.append(line)

      # Find the closest line on either side of the label
      left_spine_bound = findClosestLineToPoint((label_left_midpoint_x, midpoint_y), left_lines)
      right_spine_bound = findClosestLineToPoint((label_right_midpoint_x, midpoint_y), right_lines)

      return left_spine_bound, right_spine_bound
