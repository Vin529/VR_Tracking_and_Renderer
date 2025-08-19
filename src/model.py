""" Module for reading a .obj file into a stored model,
	retrieving vertices, faces, properties of that model.
	Written using only the Python standard library.
"""

from vector import Vector
from quaternion import Quaternion

import numpy as np
import copy

class Model(object):
	def __init__(self, file):
		self.vertices = []
		self.faces = []
		self.scale = np.array([1, 1, 1])
		self.rot = Quaternion(1, 0, 0, 0) 
		self.trans = np.array([0, 0, 0])

		#for use in the physics environment
		self.boundingSphereRadius = 0.4
		self.mass = 1
		self.area = 0.2
		self.velocity = np.array([0, 0, 0], dtype=np.float64)
		self.acceleration = np.array([0, 0, 0], dtype=np.float64)
		self.force = np.array([0, 0, 0], dtype=np.float64)
		self.experienceGravity = True
		self.experienceDrag = True
		self.experienceCollisions = True

		#for LOD management
		self.lodLevels = [[self.faces, self.vertices, 0]] #initialise base model as LOD 0
		self.activeLod = 0

		# Read in the file
		f = open(file, 'r')
		for line in f:
			if line.startswith('#'): continue
			segments = line.split()
			if not segments: continue

			# Vertices
			if segments[0] == 'v':
				vertex = Vector(*[float(i) for i in segments[1:4]])
				self.vertices.append(vertex)

			# Faces
			elif segments[0] == 'f':
				# Support models that have faces with more than 3 points
				# Parse the face as a triangle fan
				for i in range(2, len(segments)-1):
					corner1 = int(segments[1].split('/')[0])-1
					corner2 = int(segments[i].split('/')[0])-1
					corner3 = int(segments[i+1].split('/')[0])-1
					self.faces.append([corner1, corner2, corner3])

	def addLodLevel(self, lodModel, lodDistance):
		#lodDistance is the distance from the camera after which the LOD model will be used
		if lodDistance <= 0:
			raise ValueError("LOD distance must be positive")
		else:
			facesCopy = copy.deepcopy(lodModel.faces)
			verticesCopy = copy.deepcopy(lodModel.vertices)
			self.lodLevels.append([facesCopy, verticesCopy, lodDistance])
			self.lodLevels.sort(key=lambda x: x[2])

	def updateLod(self):
		#selects the appropriate LOD model based on distance from the camera (which is at origin)
		distanceFromCamera = np.linalg.norm(self.trans)
		selectedLodIndex = self.activeLod
		for index, (faces, vertices, lodDistance) in enumerate(self.lodLevels):
			if lodDistance <= distanceFromCamera:
				selectedLodIndex = index 
			else:
				break
		
		#if a new LOD is selected, switch to it
		if selectedLodIndex != self.activeLod:
			self.switchLod(selectedLodIndex)

	def switchLod(self, lodIndex):
		translation = self.trans
		rotation = self.rot
		scale = self.scale

		#move current LOD model back to the origin, reverse all transformations
		self.translateByXYZ(*-translation)
		self.localQuaternionRotate(rotation.conjugate())
		self.localScaleByXYZ(*1/scale)

		#store current LOD model
		self.lodLevels[self.activeLod][0] = self.faces
		self.lodLevels[self.activeLod][1] = self.vertices

		#switch to new LOD model
		self.faces = self.lodLevels[lodIndex][0]
		self.vertices = self.lodLevels[lodIndex][1]
		self.activeLod = lodIndex

		#move new LOD model to the original position
		self.localScaleByXYZ(*scale)
		self.localQuaternionRotate(rotation)
		self.translateByXYZ(*translation)

	def normalizeGeometry(self):
		maxCoords = [0, 0, 0]

		for vertex in self.vertices:
			maxCoords[0] = max(abs(vertex.x), maxCoords[0])
			maxCoords[1] = max(abs(vertex.y), maxCoords[1])
			maxCoords[2] = max(abs(vertex.z), maxCoords[2])

		s = 1/max(maxCoords)
		# s=1
		for vertex in self.vertices:
			vertex.x = vertex.x * s
			vertex.y = vertex.y * s
			vertex.z = vertex.z * s

	def translateByXYZ(self, xTranslation, yTranslation, zTranslation):
		translationMatrix = np.array([
			[1, 0, 0, xTranslation],
			[0, 1, 0, yTranslation],
			[0, 0, 1, zTranslation],
			[0, 0, 0, 1]
		])

		#update model position
		newPositionHomogeneous = np.dot(translationMatrix, np.array([self.trans[0], self.trans[1], self.trans[2], 1]))
		self.trans = newPositionHomogeneous[:3]
		#translate vertices
		for vertex in self.vertices:
			homogeneousVertex = np.array([vertex.x, vertex.y, vertex.z, 1])
			vertex.x, vertex.y, vertex.z, _ = translationMatrix.dot(homogeneousVertex)
	
	def localQuaternionRotate(self, rotationQuaternion: Quaternion):
		translationFromOrigin = self.trans
		#translate to origin
		self.translateByXYZ(*-translationFromOrigin)

		#update model rotation, pre multiplication as described in the slides (got this wrong initially)
		self.rot = rotationQuaternion * self.rot
		#rotate vertices
		rotationQuaternionConjugate = rotationQuaternion.conjugate()
		for vertex in self.vertices:
			#this is different to how it says in the course book, it suggests [x, y, z, 1], Not sure why
			vertexQuaternion = Quaternion(0, vertex.x, vertex.y, vertex.z)
			rotatedVertexQuaternion = rotationQuaternion * vertexQuaternion * rotationQuaternionConjugate
			vertex.x, vertex.y, vertex.z = rotatedVertexQuaternion.x, rotatedVertexQuaternion.y, rotatedVertexQuaternion.z

		#translate back to original position
		self.translateByXYZ(*translationFromOrigin)

	def globalRotateByXYZ(self, xAngle, yAngle, zAngle):
		xRotationMatrix = np.array([
        	[1, 0, 0, 0],
        	[0, np.cos(xAngle), -np.sin(xAngle), 0], 
        	[0, np.sin(xAngle), np.cos(xAngle), 0],
        	[0, 0, 0, 1]
    	])

		yRotationMatrix = np.array([
			[np.cos(yAngle), 0, np.sin(yAngle), 0],
			[0, 1, 0, 0],
			[-np.sin(yAngle), 0, np.cos(yAngle), 0],
			[0, 0, 0, 1]
		])

		zRotationMatrix = np.array([
			[np.cos(zAngle), -np.sin(zAngle), 0, 0],
			[np.sin(zAngle), np.cos(zAngle), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]
		])		

		#yaw-pitch-roll
		rotationMatrix = zRotationMatrix.dot(yRotationMatrix).dot(xRotationMatrix)
		#update model orientation
		self.rot = Quaternion.fromEulerAngles(xAngle, yAngle, zAngle) * self.rot
		#update model position, rotation when the model is not at the origin will affect it's position
		newPositionHomogeneous = np.dot(rotationMatrix, np.array([self.trans[0], self.trans[1], self.trans[2], 1]))
		self.trans = newPositionHomogeneous[:3]
		#rotate vertices
		for vertex in self.vertices:
			homogeneousVertex = np.array([vertex.x, vertex.y, vertex.z, 1])
			vertex.x, vertex.y, vertex.z, _ = rotationMatrix.dot(homogeneousVertex)

	def localRotateByXYZ(self, xAngle, yAngle, zAngle):
		translationFromOrigin = self.trans
		#translate to origin
		self.translateByXYZ(*-translationFromOrigin)

		self.globalRotateByXYZ(xAngle, yAngle, zAngle)

		#translate back to original position
		self.translateByXYZ(*translationFromOrigin)

	def globalScaleByXYZ(self, xScale, yScale, zScale):
		scaleMatrix = np.array([
			[xScale, 0, 0, 0],
			[0, yScale, 0, 0],
			[0, 0, zScale, 0],
			[0, 0, 0, 1]
		])

		#update model scale
		newScaleHomogeneous = np.dot(scaleMatrix, np.array([self.scale[0], self.scale[1], self.scale[2], 1]))
		self.scale = newScaleHomogeneous[:3]
		#update model position, scaling when the model is not at the origin will affect it's position
		newPositionHomogeneous = np.dot(scaleMatrix, np.array([self.trans[0], self.trans[1], self.trans[2], 1]))
		self.trans = newPositionHomogeneous[:3]
		#scale vertices
		for vertex in self.vertices:
			homogeneousVertex = np.array([vertex.x, vertex.y, vertex.z, 1])
			vertex.x, vertex.y, vertex.z, _ = scaleMatrix.dot(homogeneousVertex)

	def localScaleByXYZ(self, xScale, yScale, zScale):
		translationFromOrigin = self.trans
		#translate to origin
		self.translateByXYZ(*-translationFromOrigin)

		self.globalScaleByXYZ(xScale, yScale, zScale)

		#translate back to original position
		self.translateByXYZ(*translationFromOrigin)				

	################################## potential code for vectorised speedup below, not the bottleneck so not needed
	# npVertices = self.vertices_to_numpy()
	# homogeneousVertices = np.hstack((npVertices, np.ones((len(self.vertices), 1))))
	# translatedVertices = homogeneousVertices.dot(translationMatrix.T)
	# self.numpy_to_vertices(translatedVertices[:, :3])
	#		
	# def vertices_to_numpy(self):
	# 	#converts a list of Vector objects to a numpy array
	# 	return np.array([[vertex.x, vertex.y, vertex.z] for vertex in self.vertices])
    #
	# def numpy_to_vertices(self, npArray):
	# 	#converts a numpy array to a list of Vector objects
	# 	self.vertices = [Vector(x, y, z) for x, y, z in npArray]
	###########################################################################################################
