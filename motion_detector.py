# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import cv


min_h = 50
min_s = 148
min_v = 10
max_h = 250
max_s = 256
max_v = 206

hsv_min = cv.Scalar(0, 30, 20, 0)
hsv_max = cv.Scalar(15, 230, 255, 0)

def update():
	hsv_min = cv.Scalar(min_h, min_s, min_v, 0)
	hsv_max = cv.Scalar(max_h, max_s, max_v, 0)

def onminh(val):
	min_h = val
	#update()

def onmins(val):
	min_s= val
	#update()

def onminv(val):
	min_v = val
	#update()

def onmaxh(val):
	max_h = val
	#update()
#
def onmaxs(val):
	max_s = val
	#update()

def onmaxv(val):
	max_v = val
	#update()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None
i = 0

cv.NamedWindow("slider")
cv.CreateTrackbar("min_h", "slider", min_h, 255, onminh)
cv.CreateTrackbar("min_s", "slider", min_s, 255, onmins)
cv.CreateTrackbar("min_v", "slider", min_v, 255, onminv)
cv.CreateTrackbar("max_h", "slider", max_h, 255, onmaxh)
cv.CreateTrackbar("max_s", "slider", max_s, 255, onmaxs)
cv.CreateTrackbar("max_v", "slider", max_v, 255, onmaxv)

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	i += 1
	(grabbed, frame) = camera.read()
	if i % 10 != 0:
		continue

	# pre-smoothing improves Hough detector

	text = "Unoccupied"

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	frame = frame[10:410, 80:410]
	frame = cv2.flip(frame, 1)

	gray = frame - cv2.erode(frame, None)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	hsv_frame = cv2.cvtColor(frame, cv.CV_BGR2HSV)
	thresholded = cv2.inRange(hsv_frame, hsv_min, hsv_max)

	cv2.imwrite("data/%d.png" % i, thresholded)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	# loop over the contours
	#for c in cnts:
		# if the contour is too small, ignore it
		#if cv2.contourArea(c) < args["min_area"]:
			#continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		#(x, y, w, h) = cv2.boundingRect(c)
		#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		#text = "Occupied"

	# draw the text and timestamp on the frame
	#cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	#cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		#(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresholded)
	#cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()