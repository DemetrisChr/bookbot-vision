# Based on code from https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from findSpine import findSpineBoundaries
from utils import Rectangle, Line

class LabelTracker:
    def __init__(self, camera_index, trackerType, webCam, video):
        self.tracker = None
        # initialize the bounding box coordinates of the object we are going
        # to track
        self.initBB = None
        self.vs = None
        # initialize the FPS throughput estimator
        self.fps = None
        self.trackerType = args["tracker"]
        self.webCam = args.get("video", False)
        self.video = args["video"]
        self.camera_index = camera_index

    def trackLabel(self, label):
        self.setUp()
        self.track(label)



    def setUp(self):
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


        # if a video path was not supplied, grab the reference to the web cam
        if not self.webCam:
            print("[INFO] starting video stream...")
            self.vs = VideoStream(src=self.camera_index).start()
            time.sleep(1.0)
        # otherwise, grab a reference to the video file
        else:
            self.vs = cv2.VideoCapture(self.video)

    def track(self, label):
        # loop over frames from the video stream
        while True:
            # grab the current frame, then handle if we are using a
            # VideoStream or VideoCapture object
            frame = self.vs.read()
            frame = frame[1] if self.webCam else frame
            # check to see if we have reached the end of the stream
            if frame is None:
                break
            # resize the frame (so we can process it faster) and grab the
            # frame dimensions
            frame = imutils.resize(frame, width=500)
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
                    label_ractangle = Rectangle(x,y,w,h)
                    left_spine_bound, right_spine_bound = findSpineBoundaries(frame, label_ractangle)

                    #TODO: Call adjust robot with left_spine_bound and right_spine_bound to center the book in the frame

                    # Plot the lines
                    left_spine_bound.plotOnImage(frame, thickness=2)
                    right_spine_bound.plotOnImage(frame, thickness=2)

                    # Draw the rectangle around the label
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                        (0, 255, 0), 2)

                # update the FPS counter
                self.fps.update()
                self.fps.stop()

                # initialize the set of information we'll be displaying on
                # the frame
                info = [
                    ("Tracker", self.trackerType),
                    ("Success", "Yes" if success else "No"),
                    ("FPS", "{:.2f}".format(self.fps.fps())),
                ]
                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the 's' key is selected, we are going to "select" a bounding
            # box to track
            if key == ord("s"):
                # select the bounding box of the object we want to track (make
                # sure you press ENTER or SPACE after selecting the ROI)
                cv2.imwrite("frame.jpg", frame)
                self.initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                    showCrosshair=True)
                # start OpenCV object tracker using the supplied bounding box
                # coordinates, then start the FPS throughput estimator as well
                self.tracker.init(frame, self.initBB)
                self.fps = FPS().start()

                    # if the `q` key was pressed, break from the loop
            elif key == ord("q"):
                break
        # if we are using a webcam, release the pointer
        if not self.webCam:
            self.vs.stop()
        # otherwise, release the file pointer
        else:
            self.vs.release()
        # close all windows
        cv2.destroyAllWindows()

def controller_center_book(label_rectangle, camera_index):
    """
    Takes the rectangle around the label and
    will adjust the robots position until the book is in the center of the frame
    Returns the success of tracking the label
    """
    lt = LabelTracker(camera_index, "kcf", True, None)
    lt.trackLabel(label)


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
        help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="kcf",
        help="OpenCV object tracker type")
    args = vars(ap.parse_args())

    trackerType = args["tracker"]
    webCam = args.get("video", False)
    video = args["video"]

    # (x, y, width, height)
    label = (47, 140, 28, 61)

    lt = LabelTracker(0, trackerType, webCam, video)
    lt.trackLabel(label)
