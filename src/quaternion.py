import numpy as np


class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def fromGyroscopeReadings(cls, xRotationalRate, yRotationalRate, zRotationalRate, timeStep):
        angularVelocityMagnitude = np.linalg.norm([xRotationalRate, yRotationalRate, zRotationalRate])
        rotationAngle = angularVelocityMagnitude * timeStep

        if angularVelocityMagnitude == 0:
            return cls(1, 0, 0, 0)
        else:
            unitRotationAxis = [xRotationalRate, yRotationalRate, zRotationalRate] / angularVelocityMagnitude
            return cls.fromAxisAngle(unitRotationAxis, rotationAngle)
    
    @classmethod
    def fromAxisAngle(cls, axis, angle):
        axis = np.array(axis)
        if axis.size != 3:
            raise ValueError("Axis must be a 3D vector")
        
        #normalise the axis
        unitAxis = axis / np.linalg.norm(axis)

        #equation 3.30 from the course book
        w = np.cos(angle / 2)
        x, y, z = np.sin(angle / 2) * unitAxis
        return cls(w, x, y, z)

    @classmethod
    def fromEulerAngles(cls, xAngle, yAngle, zAngle):
        #method from https://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/index.htm
        #xAngle is roll, yAngle is pitch, zAngle is yaw
        cr = np.cos(xAngle / 2)
        sr = np.sin(xAngle / 2)
        cp = np.cos(yAngle / 2)
        sp = np.sin(yAngle / 2)
        cy = np.cos(zAngle / 2)
        sy = np.sin(zAngle / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return cls(w, x, y, z)
    
    def toEulerAngles(self):
        #method from https://graphics.fandom.com/wiki/Conversion_between_quaternions_and_Euler_angles
        #yaw-pitch-roll
        xAngle = np.arctan2(2 * (self.w * self.x + self.y * self.z), 1 - 2 * (self.x**2 + self.y**2))

        arcsinInput = 2 * (self.w * self.y - self.z * self.x)
        arcsinInputClipped = np.clip(arcsinInput, -1, 1)
        yAngle = np.arcsin(arcsinInputClipped)

        zAngle = np.arctan2(2 * (self.w * self.z + self.x * self.y), 1 - 2 * (self.y**2 + self.z**2))

        return xAngle, yAngle, zAngle

    def toAxisAngle(self):
        angle = 2 * np.arccos(self.w)
        
        if angle == 0:
            return np.array([1, 0, 0]), 0
        else:
            axis = np.array([self.x, self.y, self.z]) / np.sin(angle / 2)
            return axis, angle

    def conjugate(self):
        #return the conjugate of the quaternion, figure 3.12 from the course book
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def normalise(self):
        magnitude = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if magnitude > 0:
            self.w /= magnitude
            self.x /= magnitude
            self.y /= magnitude
            self.z /= magnitude
        return self
    
    def toArray(self):
        return np.array([self.w, self.x, self.y, self.z])
    
    def __mul__(self, other):
        #quaternion multiplication, equation 3.32 from course book
        w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
        x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
        y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
        z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        return Quaternion(w, x, y, z)

    def __str__(self):
        return f"w: {self.w}, x: {self.x}, y: {self.y}, z: {self.z}"         
