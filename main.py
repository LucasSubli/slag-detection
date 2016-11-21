# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
from pyimagesearch import imutils
import numpy as np
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())


# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'slag'
lower = np.array([0, 30, 0], dtype="uint8")
upper = np.array([50, 255, 150], dtype="uint8")

camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
previousFrame = None

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()
	# time.sleep(0.033)
	text = "Sem Escoria"

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if previousFrame is None:
		previousFrame = gray
		continue

	flow2 = None



	flow = cv2.calcOpticalFlowFarneback(previousFrame, gray, flow2, 0.5, 8, 15, 3, 7, 1.5, 0)

	hsv = np.zeros_like(frame)
	hsv[...,1] = 255
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	mask = cv2.inRange(rgb, lower, upper)
	mask[0:325,...] = 0;
	output = cv2.bitwise_and(rgb, rgb, mask = mask)


	cv2.imshow("mask", np.hstack([rgb, output]))

	# compute the absolute difference between the current frame and
	# first fram
	thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	(im2, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Escorrendo"

	# draw the text and timestamp on the frame
	cv2.putText(frame, "Quantidade: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
	cv2.imshow("Camera", frame)
	cv2.imshow("Escoria", thresh)
	# cv2.imshow("Frame Delta", vert)
	key = cv2.waitKey(1) & 0xFF

	previousFrame = gray

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()