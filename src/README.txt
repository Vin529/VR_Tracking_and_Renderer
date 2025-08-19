Upon running render.py, the first sequence shows the scene with the central headset rotating based off only the gyroscope and 
accelerometer data. The second sequence will start immediately after and shows the central headset rotating based off the 
Madgwick filter, which combines gyroscope, accelerometer and magnetometer readings. 

Between each sequence the renderer will pause. Pressing escape while it is paused will terminate the program.
Pressing any other key will continue onto the second sequence.

By default, the live render will only show every 10th frame, this is because rendering all 6959 frames takes
over 34 minutes on my machine. To change this behaviour, adjust the framesPerRender variable in render.py.
framesPerRender = 1 will cause every frame to be rendered in the live view.

framesPerPause can be set to allow for early termination of the live render. For example, setting framesPerPause = 1000
will stop the sequence every 1000 frames. Pressing escape while the live view is paused will terminate the program.
Pressing any other key will continue the render.

At the start of the main loop, the following variables are defined:
useMadgwickFilter
useTiltCorrection = True
useYawCorrection = False
useLodSystem = True

each of these takes a boolean value, and can be used to toggle their respective technologies in the render.
setting useMadgwickFilter = True will override any value of useTiltCorrection and useYawCorrection.

Several constants such as gravity, and coefficient of restitution, can all be adjusted by simply changing the 
properties at the top of the physicsEnvironment class in physicsEnvironment.py

to add a new model to the render:
    modelName = Model('PATH TO MODEL OBJECT FILE')
    modelName.normalizeGeometry()
    modelName.addLodLevel(LOD MODEL, LOD SWITCH DISTANCE)

    You can then change the mass, velocity, or any other physical property from the model class.
    Perform any translations, rotations or scaling required.

    modelRenderList.append(modelName)
    lodUpdateList.append(modelName)
    physicsEnvironment.addObject(modelName)