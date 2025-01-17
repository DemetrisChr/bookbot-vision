# Based on code from https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from findSpine import findSpineBoundaries, findBookBoundaries
from utils import Rectangle
from move_robot import MoveRobot
from threading import Thread


class CameraVideoStream:
    def __init__(self, camera_idx):
        """
        Initialises the camera stream and reads the first frame
        """
        self.stream = cv2.VideoCapture(camera_idx)
        self.grabbed, self.frame = self.stream.read()
        self.boundary_lines = findBookBoundaries(self.frame)
        self.stopped = False

    def start(self):
        """
        Starts the thread to read frames from the video stream
        """
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """
        Loops indefinitely and reads frames until the thread is stopped
        """
        while True:
            if self.stopped:
                return
            else:
                self.grabbed, self.frame = self.stream.read()
                self.boundary_lines = findBookBoundaries(self.frame)

    def read(self):
        """
        Returns the frame most recently read
        """
        return self.frame, self.boundary_lines

    def stop(self):
        """
        Stops the thread
        """
        self.stopped = True


class LabelTracker:
    def __init__(self, camera_index, trackerType, webCam, video=None):
        self.tracker = None

        # initialize the bounding box coordinates of the object we are going to track
        self.initBB = None
        self.vs = None

        # initialize the FPS throughput estimator
        self.fps = None
        self.trackerType = trackerType
        self.webCam = webCam
        self.video = video
        self.camera_index = camera_index
        self.video_stream = CameraVideoStream(camera_index)

    def trackLabel(self, label, debug=False):
        """
        Sets up and tracks a given label
        """
        self.setUp(debug)
        return self.track(label, debug)

    def setUp(self, debug=False):
        """
        Set up everything for the video stream and tracking
        """
        # extract the OpenCV version info
        (major, minor) = cv2.__version__.split(".")[:2]
        # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
        # function to create our object tracker
        if int(major) == 3 and int(minor) < 3:
            self.tracker = cv2.Tracker_create(trackerType.upper())
        # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
        # approrpiate object tracker constructor:
        else:
            # initialize a dictionary that maps strings to their corresponding
            # OpenCV object tracker implementations
            OPENCV_OBJECT_TRACKERS = {
                "csrt": cv2.TrackerCSRT_create,
                "kcf": cv2.TrackerKCF_create,
                "boosting": cv2.TrackerBoosting_create,
                "mil": cv2.TrackerMIL_create,
                "tld": cv2.TrackerTLD_create,
                "medianflow": cv2.TrackerMedianFlow_create,
                "mosse": cv2.TrackerMOSSE_create
            }
            # grab the appropriate object tracker using our dictionary of
            # OpenCV object tracker objects
            self.tracker = OPENCV_OBJECT_TRACKERS[self.trackerType]()

        # If using the webcam, grab the reference to the web cam
        if self.webCam:
            print("[INFO] starting video stream...")
            """
            self.vs = VideoStream(src=self.camera_index).start()
            time.sleep(1.0)
            """
            self.video_stream.start()

        # otherwise, grab a reference to the video file
        else:
            self.vs = cv2.VideoCapture(self.video)

        # Open the window
        if debug:
            cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)

    def track(self, label, debug=False):
        """
        Takes a label and tracks it in a video or webcam stram.
        Displays the video with the tracked objects.
        Returns false if the label is lost
        """
        prev_speed = 0
        count_frames_speed_0 = 0
        mv = MoveRobot()
        # loop over frames from the video stream
        while True:
            # grab the current frame, then handle if we are using a
            # VideoStream or VideoCapture object
            """
            frame = self.vs.read()
            frame = frame[1] if not self.webCam else frame
            """
            if count_frames_speed_0 >= 10:
                break
            frame, boundary_lines = self.video_stream.read()
            # check to see if we have reached the end of the stream
            if frame is None:
                break
            # resize the frame (so we can process it faster) and grab the
            # frame dimensions
            frame = imutils.resize(frame, width=500) if not self.webCam else frame
            (H, W) = frame.shape[:2]

            # Start tracking the given label
            if self.initBB is None and label is not None:
                self.initBB = label
                # start OpenCV object tracker using the supplied bounding box
                # coordinates, then start the FPS throughput estimator as well
                self.tracker.init(frame, self.initBB)
                self.fps = FPS().start()
                label = None

            # check to see if we are currently tracking an object
            if self.initBB is not None:
                # grab the new bounding box coordinates of the object
                (success, box) = self.tracker.update(frame)
                # check to see if the tracking was a success
                if success:
                    (x, y, w, h) = [int(v) for v in box]

                    # Get the spine boundary lines
                    label_rectangle = Rectangle(x, y, w, h)
                    left_spine_bound, right_spine_bound = findSpineBoundaries(label_rectangle, boundary_lines)

                    # Plot the lines
                    if debug and left_spine_bound:
                        left_spine_bound.plotOnImage(frame, thickness=2)
                    if debug and right_spine_bound:
                        right_spine_bound.plotOnImage(frame, thickness=2)

                    distance_to_middle = 0

                    # If both spine bounds are in frame and found
                    if (right_spine_bound is not None) and (left_spine_bound is not None):

                        # Adjust the position of the robot
                        # Find a point on the spine boundaries which is in the middle of the frame height
                        left_spine_coordinate = left_spine_bound.calculateXgivenY(H/2)
                        right_spine_coordinate = right_spine_bound.calculateXgivenY(H/2)

                        # Find a point on the middle of the spine
                        spine_midpoint = left_spine_coordinate + (right_spine_coordinate - left_spine_coordinate) / 2

                        # Distance from the point on the middle of the spine to the middle of the frame
                        # Range 100 if spine is on the very left of the frame to -100 on the right
                        distance_to_middle = int(( (W/2 - spine_midpoint) * 100 ) / (W/2))

                    # If only one spine bound is found
                    if (right_spine_bound is None) and (left_spine_bound is not None):

                        # Distance from the point on the middle of the spine to the middle of the frame
                        # Range 100 if spine is on the very left of the frame to -100 on the right
                        left_spine_coordinate = left_spine_bound.calculateXgivenY(H/2)
                        distance_to_middle = int(( (W/2 - left_spine_coordinate) * 100 ) / (W/2))

                    if (right_spine_bound is not None) and (left_spine_bound is None):

                        # Distance from the point on the middle of the spine to the middle of the frame
                        # Range 100 if spine is on the very left of the frame to -100 on the right
                        right_spine_coordinate = right_spine_bound.calculateXgivenY(H/2)
                        distance_to_middle = int(( (W/2 - right_spine_coordinate) * 100 ) / (W/2))

                    if (right_spine_bound is not None) or (left_spine_bound is not None):
                        if abs(distance_to_middle < 20):
                            abs_speed = 0.005
                        else:
                            abs_speed = 0.001
                        if abs(distance_to_middle) < 5:
                            speed = 0
                            count_frames_speed_0 += 1
                        elif distance_to_middle < 0:
                            speed = abs_speed
                            count_frames_speed_0 = 0
                        else:
                            speed = -abs_speed
                            count_frames_speed_0 = 0
                        if speed != prev_speed:
                            print("Moving with speed " + str(speed) + " !")
                            Thread(target=mv.setSpeed, args=(speed,)).start()
                        prev_speed = speed
                    # Draw the rectangle around the label
                    if debug:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #else:
                    #return success

                # update the FPS counter
                self.fps.update()
                self.fps.stop()
                print("FPS", "{:.2f}".format(self.fps.fps()))
                # initialize the set of information we'll be displaying on
                # the frame
                info = [
                    ("Tracker", self.trackerType),
                    ("Success", "Yes" if success else "No"),
                    ("FPS", "{:.2f}".format(self.fps.fps())),
                ]
                # loop over the info tuples and draw them on our frame
                if debug:
                    for (i, (k, v)) in enumerate(info):
                        text = "{}: {}".format(k, v)
                        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # show the output frame
            if debug:
                cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        mv.shutDown()
        # if we are using a webcam, release the pointer
        if self.webCam:
            #self.vs.stop()
            self.video_stream.stop()
        # otherwise, release the file pointer
        else:
            self.vs.release()
        # close all windows
        cv2.destroyAllWindows()
        return True


def center_spine(label_rectangle, camera_index, debug=False):
    """
    Takes the rectangle around the label and
    will adjust the robots position until the book is in the center of the frame
    Returns False if the label is no longer trackable
    """
    lt = LabelTracker(camera_index, "csrt", True, None)
    label = label_rectangle.unpack()
    return lt.trackLabel(label, debug)


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
    args = vars(ap.parse_args())

    trackerType = args["tracker"]
    webCam = not(args.get("video", False))
    video = args["video"]

    # (x, y, width, height)
    label = (47, 140, 28, 61)

    lt = LabelTracker(0, trackerType, webCam, video)
    lt.trackLabel(label)
