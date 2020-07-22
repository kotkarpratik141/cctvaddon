from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sources import configurations as config
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
from playsound import playsound



def detect_and_predict_mask(frame, faceNet, maskNet, net, ln, personIdx=0):
	(h, w) = frame.shape[:2]
	results = []
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()

	blob2 = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob2)
	layerOutputs = net.forward(ln)

	faces = []
	locs = []
	preds = []
	boxes = []
	centroids = []
	confidences = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personIdx and confidence > config.MIN_CONF:
				box = detection[0:4] * np.array([w, h, w, h])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, config.MIN_CONF, config.NMS_THRESH)

	if len(idxs) > 0:
		for i in idxs.flatten():

			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			r = (locs, preds, confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	return (r,results)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,default="mask_detector.model",help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(args["model"])

labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("Person detection is loaded")
print("proccesing input video")
#below for input video file
vs = cv2.VideoCapture("testfootage/Peoples.mp4")


#loop over video frames
while True:
	(grabbed,frame) = vs.read()
	if not grabbed:
		break
	frame = imutils.resize(frame, width=700)
	r , results= detect_and_predict_mask(frame, faceNet, maskNet,net, ln,personIdx=LABELS.index("person"))
	#print (r[0],r[1])
	violate = set()
	locs = r[0]
	preds = r[1]

	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		cv2.putText(frame, label, (startX, startY - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	if len(results) >= 2:
		centroids = np.array([r[4] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				#set mininum distance from configuration
				if D[i, j] < config.MIN_DISTANCE:
					violate.add(i)
					violate.add(j)


	for (i, (locs,preds,prob, bbox, centroid)) in enumerate(results):
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)
		if i in violate:
			color = (0, 0, 255)
		cv2.circle(frame, (cX, cY), 5, color, 2)

	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (5, frame.shape[0] - 370),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
	text1 = "Peoples detected: {}".format(len(results))
	text2 = "Peoples at risk: {}".format(len(violate))
	cv2.rectangle(frame, (0, frame.shape[0] - 400), (300, frame.shape[0] - 320), (0, 0, 0), cv2.FILLED)
	cv2.putText(frame, text1, (10, frame.shape[0] - 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
	cv2.putText(frame, text2, (10, frame.shape[0] - 335), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
	if len(violate) > 14:
		#if violation count are more than 14 it plays sound
		playsound('Sounds/alert.mp3')
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("x"):
		break
