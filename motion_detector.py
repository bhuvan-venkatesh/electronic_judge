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
import os, os.path
import sklearn
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import SVC
import collections

surf = cv2.SURF(40000)
times = 20

def getimageinfo(img):
	x_len = img.shape[1]
	kp, des = surf.detectAndCompute(img,None)
	return np.array([pt.pt[0]+pt.pt[1]*x_len for pt in kp])

def loadimage(filename=None):
	img = cv2.imread(filename)
	img = imutils.resize(img, width=500)
	img = img[10:410, 80:410]

	return getimageinfo(img)

def setsize(rayray, num=times):
	temp = np.array([0 for _ in range(num)])
	for i, e in enumerate(rayray):
		if i >= times:
			break
		temp[i] = e

	return temp


def loaddata():
	x = np.array([0 for _ in range(times)])
	y = []

	root = "data"
	for di in os.listdir(root):
		great = 0
		if int(di) > 90:
			great = 1
		for file in os.listdir(root+"/"+di):
			y.append(great)
			fullpath = root+"/"+di+"/"+file
			keyer = loadimage(filename=fullpath)
			temp = setsize(keyer)
			x = np.vstack((x, temp))

	return (x[1:], np.array(y))

def fitclassifier(classifier, x, y):
	classifier.fit(x, y)
	return accuracy_score(classifier.predict(x), y)



# loop over the frames of the video
def main(args):

	hsv_min = cv.Scalar(0, 30, 20, 0)
	hsv_max = cv.Scalar(15, 230, 255, 0)

	history = []
	max_previous = 9
	x, y = loaddata()
	print x
	print y

	if args.get("video", None) is None:
		camera = cv2.VideoCapture(0) # 0 is the webcam
		time.sleep(0.25)
	else:
		camera = cv2.VideoCapture(args["video"])

	classifier = SVC()

	print fitclassifier(classifier, x, y)

	i = 0
	while True:
		i += 1
		(grabbed, frame) = camera.read()
		if i % 10 != 0:
			continue

		# if the frame could not be grabbed, then we have reached the end
		# of the video
		if not grabbed:
			break

		# resize the frame, convert it to grayscale, and blur it
		frame = imutils.resize(frame, width=500)
		frame = frame[10:410, 80:410]
		if args.get("flip", None) is True:
			frame = cv2.flip(frame, 1)

		hsv_frame = cv2.cvtColor(frame, cv.CV_BGR2HSV)
		thresholded = cv2.inRange(hsv_frame, hsv_min, hsv_max)

		stuff = np.array([0 for _ in range(times)])


		prediction = classifier.predict(setsize(stuff))
		
		history.append(prediction)

		if len(history) > max_previous:
			history.pop()

		if prediction == 0:
			print "Score Greater than 80"
			#cv2.putText(frame, "Score Greater than 80", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1);
		else:
			print "Score Less than 80"
			cv2.putText(frame, "Score Less than 80", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1);

		# show the frame and record if the user presses a key
		cv2.imshow("Real Feed", frame)
		cv2.imshow("Threshed", thresholded)
		#cv2.imshow("Frame Delta", frameDelta)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the lop

		if key == ord("q"):
			break

	# cleanup the camera and close any open windows
	camera.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video", help="path to the video file")
	ap.add_argument("-f", "--flip", action="store_false")
	args = vars(ap.parse_args())
	main(args)


"""
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
"""