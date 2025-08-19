from model import Model

import numpy as np


class PhysicsEnvironment:
    def __init__(self):
        self.objects = []
        self.gravity = np.array([0, -9.81, 0])
        self.airDensity = 1.3
        self.dragCoefficient = 0.5
        self.timeStep = 1 / 256
        self.coefficientOfRestitution = 0.5

    def addObject(self, obj):
        if isinstance(obj, Model):
            self.objects.append(obj)
        else:
            raise ValueError("Object must be a Model")

    def update(self):
        #check for collisions
        self.checkCollisions()

        for obj in self.objects:
            #reset the force
            obj.force = np.array([0, 0, 0], dtype=np.float64)

            #apply gravity
            if obj.experienceGravity:
                obj.force += obj.mass * self.gravity

            #apply air resistance
            if obj.experienceDrag:
                #direction of drag force is opposite to velocity
                dragForce = self.dragCoefficient * (self.airDensity / 2) * np.linalg.norm(obj.velocity) * -obj.velocity  * obj.area
                obj.force += dragForce

            #update the acceleration, avoid divide by zero error
            if obj.mass == 0:
                obj.acceleration = np.array([0, 0, 0], dtype=np.float64)
            else:
                obj.acceleration = obj.force / obj.mass

            #update the velocity
            obj.velocity += obj.acceleration * self.timeStep

            #update the position
            displacement = obj.velocity * self.timeStep
            obj.translateByXYZ(displacement[0], displacement[1], displacement[2])

    def checkCollisions(self):
        #avoids checking each collision twice
        n = len(self.objects)
        for i in range(n):
            for j in range(i + 1, n):
                obj1 = self.objects[i]
                obj2 = self.objects[j]
                centreSeparation = np.linalg.norm(obj1.trans - obj2.trans)
                collisionThreshold = obj1.boundingSphereRadius + obj2.boundingSphereRadius
                if centreSeparation < collisionThreshold:
                    self.collide(obj1, obj2)

    #change the velocities of two colliding objects accordingly
    def collide(self, obj1, obj2):
        u1 = obj1.velocity
        u2 = obj2.velocity
        m1 = obj1.mass
        m2 = obj2.mass
        r1 = obj1.trans
        r2 = obj2.trans

        #immovable objects effectively have infinite mass for collision purposes
        if obj1.experienceCollisions == False:
            m1 = np.inf
        if obj2.experienceCollisions == False:
            m2 = np.inf

        #avoid division by zero error: objects with zero mass will not collide
        if obj1.mass == 0 or obj2.mass == 0:
            return

        #objects with infinite mass will not collide with each other, they will also not move during collisions with other objects
        if np.isinf(m1) and np.isinf(m2):
            return
        elif np.isinf(m1):
            massRatio1 = 1
            massRatio2 = 0
        elif np.isinf(m2):
            massRatio1 = 0
            massRatio2 = 1
        elif m1 + m2 == 0:
            #this is a bit of fun, antimatter colliding with equivalent matter will "annihilate" each other
            obj1.translateByXYZ(0, 0, 1000)
            obj2.translateByXYZ(0, 0, 1000)
            massRatio1 = 0.5
            massRatio2 = 0.5
        else:
            massRatio1 = m1 / (m1 + m2)
            massRatio2 = m2 / (m1 + m2)

        #equation from https://youtube.com/watch?v=zJLTEt_JFYg
        v1 = u1 + (1 + self.coefficientOfRestitution) * massRatio2 * (np.dot(u2 - u1, r1 - r2) / np.linalg.norm(r1 - r2)**2) * (r1 - r2)
        v2 = u2 + (1 + self.coefficientOfRestitution) * massRatio1 * (np.dot(u1 - u2, r2 - r1) / np.linalg.norm(r2 - r1)**2) * (r2 - r1)
        obj1.velocity = v1
        obj2.velocity = v2

        #move the objects apart so they are no longer colliding
        collisionDepth = obj1.boundingSphereRadius + obj2.boundingSphereRadius - np.linalg.norm(r2 - r1)
        correctionDirection = (r2 - r1) / np.linalg.norm(r2 - r1)
        #correctionVector points from obj1 to obj2
        correctionVector = collisionDepth * correctionDirection
        #move each objects in opposite directions, bigger masses move less
        obj1CorrectionVector = -correctionVector * (1 - massRatio1)
        obj2CorrectionVector = correctionVector * (1 - massRatio2)
        obj1.translateByXYZ(*obj1CorrectionVector)
        obj2.translateByXYZ(*obj2CorrectionVector)
