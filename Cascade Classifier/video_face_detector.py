from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str,
    default="data/haarcascades/haarcascade_frontalface_default.xml",
    help="path to haar cascade face detector")
args = vars(ap.parse_args())

# load the haar cascade face detector from
print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frame from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # perform face detection
    rects = detector.detectMultiScale(gray, scaleFactor=1.05,
    minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # loop over the bounding boxes
    for (x,y,w,h) in rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()