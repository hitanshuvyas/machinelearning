# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import sys, webbrowser

if len(sys.argv) > 1:       # Argument passed
    map_string = "Shri S'ad Vidya Mandal Institute Of Technology".join(sys.argv[1:])
    webbrowser.open('https://www.google.com/maps/place/' + map_string)
      
else:
   print("Pass the string as command line argument, Try Again")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

distance = 0
distance2 = 0

def draw_grid(img, grid_shape, color=(0,255,0), thickness=1):
	h, w, _ = img.shape
	rows, cols = (15,15)
	dy, dx = h / rows, w / cols

	for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
		x = int(round(x))
		cv2.line(img, (x, 0), (x, h), color=color, thickness=1)

	for y in np.linspace(start=dy, stop=w-dy, num=cols-1):
		y = int(round(y))
		cv2.line(img, (0, y), (w, y), color=color, thickness=1)

	return img


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net2 = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
vs2 = VideoStream(src=1).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame2 = vs2.read()
	frame = imutils.resize(frame, width=600, height=600)
	frame2 = imutils.resize(frame2, width=600, height=600)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	(h, w) = frame2.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
	blob2 = cv2.dnn.blobFromImage(cv2.resize(frame2, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	net2.setInput(blob2)
	detections = net.forward()
	detections2 = net2.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2], detections2.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
		confidence2 = detections2[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"] or confidence2 > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			idx2 = int(detections2[0, 0, i, 1])
			box2 = detections2[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX2, startY2, endX2, endY2) = box2.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)

			label2 = "{}: {:.2f}%".format(CLASSES[idx2],
				confidence2 * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			if(CLASSES[idx] == "person") :
				boxLength = endY-startY
				noOfBox = round(boxLength/15)
				if(noOfBox > 1) :
					distance = 39.098/noOfBox
				else :
					distance = 34.6
				cv2.putText(frame, f'Distance: {distance}', (40,70), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 2)
				
			cv2.rectangle(frame2, (startX2, startY2), (endX2, endY2),
				COLORS[idx2], 2)
			y2 = startY2 - 15 if startY2 - 15 > 15 else startY2 + 15
			if(CLASSES[idx2] == "person") :
				boxLength2 = endY2-startY2
				noOfBox2 = round(boxLength2/15)
				if(noOfBox > 1) :
					distance2 = 39.098/noOfBox2
				else :
					distance2 = 34.6
				cv2.putText(frame2, f'Distance: {distance2}', (40,70), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,0,0), 2)

			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			cv2.putText(frame2, label2, (startX2, y2),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx2], 2)

	# show the output frame
	draw_grid(frame, (300,300))
	draw_grid(frame2, (300,300))
	Hori = np.concatenate((frame, frame2), axis=1)
	cv2.imshow("Camera Feeds", Hori)
	#cv2.imshow("Frame1", frame)
	#cv2.imshow("Frame2", frame2)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
vs2.stop()

# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
