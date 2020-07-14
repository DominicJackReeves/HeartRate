# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import sys
from matplotlib.path import Path


def get_roi(polygon):
	justy = np.array(polygon)[:,1]
	justx = np.array(polygon)[:,0]
	miny = justy.min()
	minx = justx.min()
	maxy = justy.max()
	maxx = justx.max()
	poly = []
	for element in polygon:
		poly.append((element[0], element[1]))
	x, y = np.meshgrid(np.arange(minx, maxx), np.arange(miny, maxy)) # make a canvas with coordinates
	x, y = x.flatten(), y.flatten()
	points = np.vstack((x,y)).T
	p = Path(poly) # make a polygon
	grid = p.contains_points(points)
	grid = np.vstack((grid, grid)).T
	grid = np.invert(grid)
	masked_points = np.ma.masked_array(points, grid)
	return masked_points
	

def get_average_in_roi(masked_points, image):
	values = image[masked_points[:,1].compressed(), masked_points[:,0].compressed(), :]
	means = np.mean(values, 0)
	return means


def initiate(image):
	# construct the argument parser and parse the arguments
	#ap = argparse.ArgumentParser()
	#ap.add_argument("-i", "--image", required=True,
	#	help="path to input image")
	#args = vars(ap.parse_args())

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = []
	detector.append(dlib.get_frontal_face_detector())
	detector.append(cv2.CascadeClassifier("haarcascade_frontalface_default.xml"))
	detector.append(cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml"))
	detector.append(cv2.CascadeClassifier("haarcascade_frontalface_alt.xml"))
	detector.append(cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml"))
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	# load the input image, resize it, and convert it to grayscale
	#image = cv2.imread(image)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector[0](gray, 1)
	count = 1
	while rects == [] and count < 5:
		rects = detector[count].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
		count += 1
	average = [0,0,0]
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		#Determining region of interest from face.

		polygon = [shape[1], shape[4], shape[14], shape[17]]
		points = get_roi(polygon)
		if len(points) > 0:
			average = get_average_in_roi(points,image)


	return average




if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	args = vars(ap.parse_args())
	print(args)
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(args["image"])
	print(type(image))
	initiate(image)

	
                 