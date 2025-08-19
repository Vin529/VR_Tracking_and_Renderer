The code and report in this repository were submitted as part of the 3rd year Deep Learning Module to the Computer Science Department at Durham University. This work was awarded a mark of 89% (6th highest out of 100 students).

# Features
In this project, I first extended the basic Python renderer at https://github.com/ecann/RenderPy with the following functionality:
* Real time output of the frame buffer
* Perspective projection to replace the orthographic projection
* Transformation matrices for translation, rotation, and scaling of objects
* Custom quaternion class for improved model transformations

The provided headset data in "src/data/IMUData.csv" contains accelerometer, magnetometer and gyroscope readings collected from an IMU mounted to a VR headset. This data captures a sequence of rotations about each axis, made over a 27-second recording.

I implemented a dead reckoning filter for integration of gyroscope readings, as well as gravity-based tilt correction and magnetometer-based yaw correction. To address shortcomings with this approach I also implemented the Madgwick algorithm, using the C implementation (https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/) as my reference. This functions as a complementary filter that combines high-pass filtered gyroscope measurements with low-pass filtered measurements from other sensors to reduce jitter and improve tracking accuracy.

On top of this I engineered a physics system from scratch, with support for custom physical properties for different objects. This physics system can be seen powering the gravity and collisions of the headsets in the provided video sequences.

A level-of-detail (LOD) optimisation strategy was used, which swaps in lower quality meshes as each object moves further away from the camera. This significantly reduces render time with minimal loss to visual quality.

The provided videos demonstrate each of these features in action.


# Repository Structure
model.py, physicsEnvironment.py and quaternion.py are all implemented from scratch, and are entirely my own code. Other than part of the renderModel() function, render.py is also my own code. image.py, shape.py and vector.py are all largely unchanged from their original implementations in RenderPy (https://github.com/ecann/RenderPy).

Positions, weights, initial velocities and other properties for each object in the scene can be changed in render.py, as well as adding additional objects. Properties of the physics system itself, such as drag coefficient and air resistance, can be changed in physicsEnvironment.py. Detailed instructions for this can be found in "src/README.txt".

The video directory contains four 27 second renders, each showing the same scene, but with a different method used for correcting tracking error.

report.pdf contains several static renders of the headset model. There is also discussion of the various approaches used for correcting headset tracking errors, along with objective performance analysis and comparisons for each method. Implementation of the physics system and LOD optimisation are described in the report, with performance profiling and suggestions for further improvement.