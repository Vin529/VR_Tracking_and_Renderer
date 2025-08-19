from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector
from quaternion import Quaternion
from physicsEnvironment import PhysicsEnvironment

import cv2
import numpy as np
import time

#render every nth frame
FRAMES_PER_RENDER = 10
#pause every n frames, any key to continue, esc to stop
FRAMES_PER_PAUSE = 1000

width = 512
height = 512
image = Image(width, height, Color(255, 255, 255, 255))

# Init z-buffer
zBuffer = [-float('inf')] * width * height


def getVertexNormal(vertIndex, faceNormalsByVertex):
	# Compute vertex normals by averaging the normals of adjacent faces
	normal = Vector(0, 0, 0)
	for adjNormal in faceNormalsByVertex[vertIndex]:
		normal = normal + adjNormal

	return normal / len(faceNormalsByVertex[vertIndex])


def getOrthographicProjection(x, y, z):
	# Convert vertex from world space to screen space
	# by dropping the z-coordinate (Orthographic projection)
	screenX = int((x+1.0)*width/2.0)
	screenY = int((y+1.0)*height/2.0)

	return screenX, screenY


def getPerspectiveProjection(x, y, z, nearPlane=-0.5, farPlane=-150):
	#camera is looking down the negative z-axis so near and far are negative values
	#1x1 viewing window on the near plane
	left = -0.5
	right = 0.5
	bottom = -0.5
	top = 0.5

	perspectiveMatrix = np.array([
		[nearPlane, 0, 0, 0],
		[0, nearPlane, 0, 0],
		[0, 0, nearPlane + farPlane, -(farPlane * nearPlane)],
		[0, 0, 1, 0]
	])

	scaleTranslationMatrix = np.array([
		[2 / (right - left), 0, 0, -(right + left) / (right - left)],
		[0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
		[0, 0, 2 / (nearPlane - farPlane), -(farPlane + nearPlane) / (nearPlane - farPlane)],
		[0, 0, 0, 1]
	])

	viewPortMatrix = np.array([
		[width / 2, 0, 0, (width - 1) / 2],
		[0, height / 2, 0, (height - 1) / 2],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	])

	w = 1
	point = np.array([x, y, z, w])

	#perspective transform
	point = perspectiveMatrix.dot(point)
	if point[3] == 0:
		#avoid division by zero
		point[3] = 0.0001
	point = point / point[3]
	#transformedZ is used for z-buffering
	transformedZ = point[2]

	#scale and translate
	point = scaleTranslationMatrix.dot(point)

	#viewPort transform
	screenPoint = viewPortMatrix.dot(point)

	screenX = int(screenPoint[0])
	screenY = int(screenPoint[1])
	return screenX, screenY, transformedZ


def renderModel(model):
	#define the near and far planes for perspective projection and culling
	nearPlane = -0.5
	farPlane = -150

	# Calculate face normals
	faceNormals = {}
	for face in model.faces:
		p0, p1, p2 = [model.vertices[i] for i in face]
		faceNormal = (p2-p0).cross(p1-p0).normalize()

		for i in face:
			if not i in faceNormals:
				faceNormals[i] = []

			faceNormals[i].append(faceNormal)

	# Calculate vertex normals
	vertexNormals = []
	for vertIndex in range(len(model.vertices)):
		vertNorm = getVertexNormal(vertIndex, faceNormals)
		vertexNormals.append(vertNorm)

	# Render the image iterating through faces
	for face in model.faces:
		p0, p1, p2 = [model.vertices[i] for i in face]
		n0, n1, n2 = [vertexNormals[i] for i in face]

		# Define the light direction
		lightDir = Vector(0, 0, -1)

		# Set to true if face should be culled
		cull = False

		# Transform vertices and calculate lighting intensity per vertex
		transformedPoints = []
		for p, n in zip([p0, p1, p2], [n0, n1, n2]):
			intensity = n * lightDir

			# Intensity < 0 means light is shining through the back of the face
			# In this case, don't draw the face at all ("back-face culling")
			if intensity < 0:
				#the line below was included in the provided material. The comment is incorrect and the culling dosent take perspective projection into account so wouldnt work anyway
				#cull = True # Back face culling is disabled in this version
				intensity = 0 #make any surface not exposed to the light completely black

			#screenX, screenY = getOrthographicProjection(p.x, p.y, p.z)
			screenX, screenY, transformedZ = getPerspectiveProjection(p.x, p.y, p.z, nearPlane, farPlane)
			
			#cull faces that are in front of the near plane or behind the far plane
			if transformedZ > nearPlane or transformedZ < farPlane:
				cull = True

			#p.z replaced with transformedZ for perspective adjusted z-buffering
			transformedPoints.append(Point(screenX, screenY, transformedZ, Color(intensity*255, intensity*255, intensity*255, 255)))

		if not cull:
			Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw_faster(image, zBuffer)


#my own function for rendering a coloured arrow from the models centre to the given vector, useful for visualisation
def renderVector(model, vector, color: Color = Color(255, 0, 0, 255)):
		#billboard style arrow originating at the headset's centre and going to a point at the given vector
		xModelCentre, yModelCentre, zModelCentre = model.trans
		arrowVerticalPoints = [
			Point(xModelCentre, yModelCentre - 0.05, zModelCentre),
			Point(xModelCentre, yModelCentre + 0.05, zModelCentre),
			Point(xModelCentre + vector[0], yModelCentre + vector[1], zModelCentre + vector[2])
		]
		arrowHorizontalPoints = [
			Point(xModelCentre - 0.05, yModelCentre, zModelCentre),
			Point(xModelCentre + 0.05, yModelCentre, zModelCentre),
			Point(xModelCentre + vector[0], yModelCentre + vector[1], zModelCentre + vector[2])
		]

		#apply perspective projection to the arrow points
		transformedArrowVerticalPoints = []
		transformedArrowHorizontalPoints = []
		for point in arrowVerticalPoints:
			screenX, screenY, transformedZ = getPerspectiveProjection(point.x, point.y, point.z)
			transformedArrowVerticalPoints.append(Point(screenX, screenY, transformedZ, color))
		for point in arrowHorizontalPoints:
			screenX, screenY, transformedZ = getPerspectiveProjection(point.x, point.y, point.z)
			transformedArrowHorizontalPoints.append(Point(screenX, screenY, transformedZ, color))

		#draw the arrow
		Triangle(transformedArrowVerticalPoints[0], transformedArrowVerticalPoints[1], transformedArrowVerticalPoints[2]).draw_faster(image, zBuffer)
		Triangle(transformedArrowHorizontalPoints[0], transformedArrowHorizontalPoints[1], transformedArrowHorizontalPoints[2]).draw_faster(image, zBuffer)


def getTiltCorrectionQuaternion(model, xAcceleration, yAcceleration, zAcceleration, tiltGainCoefficient):
	localSensorUpQuaternion = Quaternion(0, xAcceleration, yAcceleration, zAcceleration)
	#rotate the headset's local up vector by the model's rotation to return to the global frame
	#lecture slides say to use inverse body rotation, dosent make sense to me. Course book says "This involves rotating it by q(t), the body orientation" so will stick with this
	globalSensorUpQuaternion = model.rot * localSensorUpQuaternion * model.rot.conjugate()
	globalSensorUpVector = np.array([globalSensorUpQuaternion.x, globalSensorUpQuaternion.y, globalSensorUpQuaternion.z])
	
	#only apply tilt correction if the magnitude of the sensor's up vector is within the tolerance percentage of 1 (g)
	tolerance = 0.02
	minimumMagnitude = 1 - tolerance
	maximumMagnitude = 1 + tolerance
	if minimumMagnitude <= np.linalg.norm(globalSensorUpVector) <= maximumMagnitude:
		globalSensorUpUnitVector = globalSensorUpVector / np.linalg.norm(globalSensorUpVector)

		# ######################################## uncomment section below to see the sensor's up vector in the render
		# print("globalSensorUpVector", globalSensorUpVector)
		# renderVector(model, 3 * globalSensorUpVector, Color(0, 0, 0, 255))
		# ######################################################################
		
		AbsoluteUpUnitVector = [0, 1, 0]
		#tilt axis is [Uz, 0, -Ux] so it's orthogonal to y axis and globalSensorUpVector, could accomplish the same with a cross product
		tiltCorrectionAxis = np.array([globalSensorUpVector[2], 0, -globalSensorUpVector[0]])
		#tilt angle has to be negated because globalSensorUpVector has to be rotated anti clockwise about the tilt axis to align with AbsoluteUpVector
		tiltCorrectionAngle = -np.arccos(np.dot(globalSensorUpUnitVector, AbsoluteUpUnitVector))
		#tilt angle error added to lsit for logging purposes, finding average deviation at the end of the render
		tiltAngleErrorList.append(np.rad2deg(abs(tiltCorrectionAngle))) 
		tiltCorrectionQuaternion = Quaternion.fromAxisAngle(tiltCorrectionAxis, tiltGainCoefficient * tiltCorrectionAngle)
		return tiltCorrectionQuaternion	
	else:
		#print("globalSensorUpVector not close enough to 1 (g) to correct tilt")
		return Quaternion(1, 0, 0, 0)


def getYawCorrectionQuaternion(model, xMagField, yMagField, zMagField, referenceNorthVector, yawGainCoefficient):
	localSensorMagFieldQuaternion = Quaternion(0, xMagField, yMagField, zMagField)
	#LavYerKatAnt14.pdf says to rotate by inverse of body rotation, this still dosent makse sense to me so I use body rotation
	globalSensorMagFieldQuaternion = model.rot * localSensorMagFieldQuaternion * model.rot.conjugate()
	sensorNorthVector = np.array([globalSensorMagFieldQuaternion.x, 0, globalSensorMagFieldQuaternion.z])	

	#only apply yaw correction if the magnitude of the sensor's north vector is within the tolerance percentage of the reference magnitude
	tolerance = 0.05
	referenceNorthVectorMagnitude = np.linalg.norm(referenceNorthVector)
	minimumMagnitude = (1 - tolerance) * referenceNorthVectorMagnitude
	maximumMagnitude = (1 + tolerance) * referenceNorthVectorMagnitude
	if minimumMagnitude <= np.linalg.norm(sensorNorthVector) <= maximumMagnitude:
		sensorNorthAngle = np.arctan2(sensorNorthVector[0], sensorNorthVector[2])
		referenceNorthAngle = np.arctan2(referenceNorthVector[0], referenceNorthVector[2]) #this dosent have to be recalculated every time, could just get passed
		yawCorrectionAngle = referenceNorthAngle - sensorNorthAngle
		yawCorrectionAxis = np.array([0, 1, 0])
		yawCorrectionQuaternion = Quaternion.fromAxisAngle(yawCorrectionAxis, yawGainCoefficient * yawCorrectionAngle)

		# ########################## uncomment section below to see the sensor's magnetic field vector in the render
		# localSensorMagFieldVector = np.array([xMagField, yMagField, zMagField])
		# globalSensorMagFieldVector = np.array([globalSensorMagFieldQuaternion.x, globalSensorMagFieldQuaternion.y, globalSensorMagFieldQuaternion.z])
		#
		# #visible sensor vectors for debugging or visualisation
		# print("localSensorVector", localSensorMagFieldVector)
		# print("globalSensorVector", globalSensorMagFieldVector)
		# print("correction angle:", np.rad2deg(yawCorrectionQuaternion.toEulerAngles()) / yawGainCoefficient)
		# renderVector(model, 3 * globalSensorMagFieldVector, Color(255, 255, 0, 255))
		# renderVector(model, 6 * referenceNorthVector)
		# renderVector(model, 6 * sensorNorthVector, Color(0, 255, 0, 255))
		# ##################################################################

		return yawCorrectionQuaternion
	else:
		#print("sensorNorthVector magnitude not close enough to reference to correct yaw")
		return Quaternion(1, 0, 0, 0)


def getMadgwickFilterQuaternion(model, rotationalRateVector, accelerationVector, magFieldVector, beta, timeStep):
	rotationalRateQuaternion = Quaternion(0, *rotationalRateVector)

	#convert model orientation into the IMU's coordinate system
	toRenderAxesQuaternion = Quaternion(-0.5, 0.5, 0.5, 0.5)
	correctedModelRot = toRenderAxesQuaternion.conjugate() * model.rot * toRenderAxesQuaternion

	#normalise acceleration and magnetic field vectors
	if np.linalg.norm(accelerationVector) == 0 or np.linalg.norm(magFieldVector) == 0:
		print("acceleration or magnetic field vector is zero, skipping Madgwick filter update")
		return Quaternion(1, 0, 0, 0)
	else:
		accelerationVector = accelerationVector / np.linalg.norm(accelerationVector)
		magFieldVector = magFieldVector / np.linalg.norm(magFieldVector)
	
    #rate of change of quaternion from gyroscope, converted to array for easier multiplication and addition
	qDotArray = (correctedModelRot * rotationalRateQuaternion).toArray() * 0.5

	#put headset data into variables to be used in following code
	q0, q1, q2, q3 = correctedModelRot.w, correctedModelRot.x, correctedModelRot.y, correctedModelRot.z
	ax, ay, az = accelerationVector
	mx, my, mz = magFieldVector

	#the following three code blocks are adapted from the original C implementation of the Madgwick filter
	#which can be found at https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/
	#auxiliary variables to avoid repeated arithmetic
	_2q0mx = 2 * q0 * mx
	_2q0my = 2 * q0 * my
	_2q0mz = 2 * q0 * mz
	_2q1mx = 2 * q1 * mx
	_2q0 = 2 * q0
	_2q1 = 2 * q1
	_2q2 = 2 * q2
	_2q3 = 2 * q3
	_2q0q2 = 2 * q0 * q2
	_2q2q3 = 2 * q2 * q3
	q0q0 = q0 * q0
	q0q1 = q0 * q1
	q0q2 = q0 * q2
	q0q3 = q0 * q3
	q1q1 = q1 * q1
	q1q2 = q1 * q2
	q1q3 = q1 * q3
	q2q2 = q2 * q2
	q2q3 = q2 * q3
	q3q3 = q3 * q3

	#reference direction of Earth's magnetic field
	hx = mx * q0q0 - _2q0my * q3 + _2q0mz * q2 + mx * q1q1 + _2q1 * my * q2 + _2q1 * mz * q3 - mx * q2q2 - mx * q3q3
	hy = _2q0mx * q3 + my * q0q0 - _2q0mz * q1 + _2q1mx * q2 - my * q1q1 + my * q2q2 + _2q2 * mz * q3 - my * q3q3
	_2bx = np.sqrt(hx * hx + hy * hy)
	_2bz = -_2q0mx * q2 + _2q0my * q1 + mz * q0q0 + _2q1mx * q3 - mz * q1q1 + _2q2 * my * q3 - mz * q2q2 + mz * q3q3
	_4bx = 2 * _2bx
	_4bz = 2 * _2bz
	
	#gradient descent algorithm corrective step
	s0 = -_2q2 * (2 * q1q3 - _2q0q2 - ax) + _2q1 * (2 * q0q1 + _2q2q3 - ay) - _2bz * q2 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (-_2bx * q3 + _2bz * q1) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + _2bx * q2 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);
	s1 = _2q3 * (2 * q1q3 - _2q0q2 - ax) + _2q0 * (2 * q0q1 + _2q2q3 - ay) - 4 * q1 * (1 - 2 * q1q1 - 2 * q2q2 - az) + _2bz * q3 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (_2bx * q2 + _2bz * q0) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + (_2bx * q3 - _4bz * q1) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);
	s2 = -_2q0 * (2 * q1q3 - _2q0q2 - ax) + _2q3 * (2 * q0q1 + _2q2q3 - ay) - 4 * q2 * (1 - 2 * q1q1 - 2 * q2q2 - az) + (-_4bx * q2 - _2bz * q0) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (_2bx * q1 + _2bz * q3) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + (_2bx * q0 - _4bz * q2) * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);
	s3 = _2q1 * (2 * q1q3 - _2q0q2 - ax) + _2q2 * (2 * q0q1 + _2q2q3 - ay) + (-_4bx * q3 + _2bz * q1) * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx) + (-_2bx * q0 + _2bz * q2) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my) + _2bx * q1 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz);
	
	#normalise s
	s = np.array([s0, s1, s2, s3])
	sNormalised = s / np.linalg.norm(s)
	
	#apply feedback step
	qDotArray = qDotArray - (beta * sNormalised)
	#Integrate rate of change of quaternion to yield new q
	qArray = correctedModelRot.toArray()
	qArray = qArray + (qDotArray * timeStep)
	#normalise q
	qArray = qArray / np.linalg.norm(qArray)

	#find the quaternion to rotate the model to the new correct orientation
	destinationQuaternion = Quaternion(qArray[0], qArray[1], qArray[2], qArray[3])
	orientationCorrectionQuaternion = destinationQuaternion * correctedModelRot.conjugate()
	#convert orientationCorrectionQuaternion to correct coordinate system for render
	orientationCorrectionQuaternion = toRenderAxesQuaternion * orientationCorrectionQuaternion * toRenderAxesQuaternion.conjugate()
	return orientationCorrectionQuaternion.normalise()


# ======================================================================================================================
# Main loop
# ======================================================================================================================

#run two sequences back to back, the first with only tilt correction and the second with Madgwick filter for both tilt and yaw
for mainLoopCounter in range(2):
	start_time = time.time()
	#used to measure average deviation from true tilt over the render. Will only print at the end if useTiltCorrection = True
	tiltAngleErrorList = []

	#run without Madgwick or yaw correction first, then with Madgwick (which replaces normal tilt and yaw correction)
	if mainLoopCounter == 0:
		#set to True to use Madgwick filter for orientation correction, False to use dead reckoning with independent tilt and yaw correction
		useMadgwickFilter = False
	else: 
		useMadgwickFilter = True

	#if Madwick filter is set to False, use these variables to toggle tilt or yaw correction on or off
	useTiltCorrection = True
	useYawCorrection = False
	#set to True to use the LOD system, False to disable it
	useLodSystem = True

	#these models are used as low LOD for far away headsets
	halfLodModel = Model('data/headset_half_poly.obj')
	halfLodModel.normalizeGeometry()
	quarterLodModel = Model('data/headset_quarter_poly.obj')
	quarterLodModel.normalizeGeometry()

	#load the main model, this is the one which will rotate using IMU data
	model = Model('data/headset.obj')
	model.normalizeGeometry()
	model.addLodLevel(halfLodModel, 7) #LOD model, followed by the distance past which it should be swapped in
	model.addLodLevel(quarterLodModel, 14)
	model.experienceGravity = False
	model.experienceCollisions = False
	model.translateByXYZ(0, 0, -3)

	#second model #################################### this is the headset which flies away into the distance at a constant speed, and under no effects from gravity or drag
	model2 = Model('data/headset.obj')
	model2.normalizeGeometry()
	model2.addLodLevel(halfLodModel, 7)
	model2.addLodLevel(quarterLodModel, 14)
	model2.experienceGravity = False
	model2.experienceDrag = False
	model2.velocity = np.array([1, 0, -10], dtype=np.float64) #set initial velocity to cause collisions
	model2.translateByXYZ(-1, 1, 25)
	model2.localRotateByXYZ(0, -np.pi/12, 0)

	#third model #################################### this is the headset which flies towards the camera from the distance at a constant speed, and under no effects from gravity or drag
	model3 = Model('data/headset.obj')
	model3.normalizeGeometry()
	model3.addLodLevel(halfLodModel, 7)
	model3.addLodLevel(quarterLodModel, 14)
	model3.experienceGravity = False
	model3.experienceDrag = False
	model3.velocity = np.array([1, 0, 10], dtype=np.float64)
	model3.translateByXYZ(-10, 1, -85)
	model3.localRotateByXYZ(0, np.pi/12, 0)

	#fourth model #################################### this is the headset which flies in from the right, with 1/10th the mass of all other models
	model4 = Model('data/headset.obj')
	model4.normalizeGeometry()
	model4.addLodLevel(halfLodModel, 7)
	model4.addLodLevel(quarterLodModel, 14)
	model4.mass = 0.1
	model4.velocity = np.array([-130, 70, 0], dtype=np.float64)
	model4.translateByXYZ(6, 2, -3)
	model4.localRotateByXYZ(0, 0, np.pi/4)

	#fifth model #################################### this is the headset which flies in from the back left
	model5 = Model('data/headset.obj')
	model5.normalizeGeometry()
	model5.addLodLevel(halfLodModel, 7)
	model5.addLodLevel(quarterLodModel, 14)
	model5.velocity = np.array([10, 9, 15], dtype=np.float64)
	model5.translateByXYZ(-9, 2, -16)
	model5.localRotateByXYZ(0, np.pi/4, -np.pi/4)

	#sixth model #################################### this is the headset which drops directly vertically onto the main model
	model6 = Model('data/headset.obj')
	model6.normalizeGeometry()
	model6.addLodLevel(halfLodModel, 10) #lod distances are higher as quality drop is more noticeable from this angle
	model6.addLodLevel(quarterLodModel, 20)
	model6.velocity = np.array([0, 8, 0], dtype=np.float64)
	model6.translateByXYZ(0, 4, -3)
	model6.localRotateByXYZ(np.pi/2, np.pi/4, 0)

	#seventh model #################################### this is the final headset which comes in from the left side, behind the camera
	#this model is not spawned in here at the start, it is spawned in the render loop after 750 frames
	model7 = Model('data/headset.obj')
	model7.normalizeGeometry()
	model7.addLodLevel(halfLodModel, 7)
	model7.addLodLevel(quarterLodModel, 14)
	model7.velocity = np.array([40, 7, -40], dtype=np.float64)
	model7.translateByXYZ(-5, 0, 2)
	model7.localRotateByXYZ(0, np.pi * 3/4, 0)

	#these models' lod will be updated in the render loop
	lodUpdateList = []
	lodUpdateList.extend([model, model2, model3, model4, model5, model6])

	#these models will be rendered in the render loop
	modelRenderList = []
	modelRenderList.extend([model, model2, model3, model4, model5, model6])

	#initialise physics environment
	physicsEnvironment = PhysicsEnvironment()
	physicsEnvironment.addObject(model)
	physicsEnvironment.addObject(model2)
	physicsEnvironment.addObject(model3)
	physicsEnvironment.addObject(model4)
	physicsEnvironment.addObject(model5)
	physicsEnvironment.addObject(model6)

	#read in IMU data
	filepath = 'data/IMUData.csv'
	IMUdata = np.genfromtxt(filepath, delimiter=',', skip_header=1)
	#convert rotational velocities from deg/sec to rad/sec
	IMUdata[:, 1:4] = np.deg2rad(IMUdata[:, 1:4])
	timeStepArray = np.diff(IMUdata[:, 0])

	#used in the calibration phase for yaw correction
	NorthVectorEstimationList = []
	referenceNorthVector = None

	#model starts at identity quaternion (1, 0, 0, 0) rotation
	for i in range(len(timeStepArray)):
		#clear last frame
		image.clear()
		zBuffer = [-float('inf')] * width * height

		timeStep = timeStepArray[i]
		IMUdataRow = IMUdata[i]

		if useMadgwickFilter:
			#Madgwick filter update
			beta = 0.05
			rotationalRateVector = IMUdataRow[1:4]
			accelerationVector = IMUdataRow[4:7]
			magFieldVector = IMUdataRow[7:10]
			deltaQuaternion = getMadgwickFilterQuaternion(model, rotationalRateVector, accelerationVector, magFieldVector, beta, timeStep)
		else:
			#dead reckoning filter
			#the IMU was mounted on the headset so that z was up. I think this is the appropriate axis correction
			#I think z was up, x was out of the front of the headset, and y went to the right if looking at headset front on. Accelerometer readings line up with this
			#so to fix: x -> z, y -> x, z -> y
			zRotationalRate, xRotationalRate, yRotationalRate = IMUdataRow[1:4]
			deltaQuaternion = Quaternion.fromGyroscopeReadings(xRotationalRate, yRotationalRate, zRotationalRate, timeStep)

			if useTiltCorrection:
				#gravity-based tilt correction
				tiltGainCoefficient = 0.02
				#apply same axis correction as with rotational rates above
				zAcceleration, xAcceleration, yAcceleration = IMUdataRow[4:7]
				tiltCorrectionQuaternion = getTiltCorrectionQuaternion(model, xAcceleration, yAcceleration, zAcceleration, tiltGainCoefficient)
				deltaQuaternion = tiltCorrectionQuaternion * deltaQuaternion

			if useYawCorrection:
				#magnetometer-based yaw correction
				yawGainCoefficient = 0.02
				northCalibrationFrames = 100
				NorthCalibrationStartFrame = 200
				YawCorrectionStartFrame = NorthCalibrationStartFrame + northCalibrationFrames
				#apply same axis correction as with rotational rates and acceleration above
				zMagField, xMagField, yMagField = IMUdataRow[7:10]
				
				#a run of frames near the start are used to establish the direction of the reference horizontal north vector
				if NorthCalibrationStartFrame <= i < YawCorrectionStartFrame:
					localSensorMagFieldQuaternion = Quaternion(0, xMagField, yMagField, zMagField)
					globalSensorMagFieldQuaternion = model.rot * localSensorMagFieldQuaternion * model.rot.conjugate()
					NorthVectorEstimation = np.array([globalSensorMagFieldQuaternion.x, 0, globalSensorMagFieldQuaternion.z])
					NorthVectorEstimationList.append(NorthVectorEstimation)
				elif i == YawCorrectionStartFrame:
					#this is the direction of the horizontal north vector in the headset's initial position, averaged for improved accuracy
					referenceNorthVector = np.mean(NorthVectorEstimationList, axis=0)

				#once we have a reference north vector to aim for, we can start correcting the yaw
				if referenceNorthVector is not None:
					yawCorrectionQuaternion = getYawCorrectionQuaternion(model, xMagField, yMagField, zMagField, referenceNorthVector, yawGainCoefficient)
					deltaQuaternion = yawCorrectionQuaternion * deltaQuaternion

		#apply rotation from IMU data to the main model
		model.localQuaternionRotate(deltaQuaternion)

		physicsEnvironment.update()

		#render every nth frame
		if i % FRAMES_PER_RENDER == 0:
			print("frame", i)
			#update lod for each model, only if useLodSystem is set to True
			if useLodSystem:
				for eachModel in lodUpdateList:
					eachModel.updateLod()

			#render each model
			for eachModel in modelRenderList:
				renderModel(eachModel)

			#save frames as PNGs
			#image.saveAsPNG(f"renderFrames_deadReckoning/frame_{i}.png")

			#show scene in window
			image.displayBuffer()
			#have to have waitKey to give time to update image
			cv2.waitKey(1)
	
		#model7 is spawned in at frame 750
		if i == 750:
			lodUpdateList.append(model7)
			modelRenderList.append(model7)
			physicsEnvironment.addObject(model7)

		#pause every n frames, any key to continue, esc to stop
		if i % FRAMES_PER_PAUSE == 0:
			key = cv2.waitKey(0)  #0 means wait indefinitely
			if key == 27:  #exit the loop if ESC is pressed
				break  

	end_time = time.time()
	print("execution time:", end_time - start_time, "seconds")

	#output average tilt angle error
	if tiltAngleErrorList:
		print("average tilt angle error:", np.mean(tiltAngleErrorList))

	cv2.destroyAllWindows()
