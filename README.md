![til](./pose.gif)

Pose Computation/Estimation
---------------------------
Perspective-n-Point problem(https://medium.com/@rashik.shrestha/perspective-n-point-pnp-f2c7dd4ef1ed) - Half of the calibration problem with camera intrinsic params given and only extrinsic params(T_cw) need to be calculated.
 - Uses the two stage Levenberg-Marquardt optimization. Initial solution for non-planar object points requires 6 points and uses DLT (Direct Linear Transformation https://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho1-21-DLT.pptx.pdf) algorithm. For planar points, intial pose requires 4 points and uses homography.

As of OpenCV 4.8, the Aruco module has been moved to the object detection library. The python documentation hasn't kept up and in some cases not all functions have been migrated, e.g. cv2.aruco.calibrateCameraCharuco.
Here, I do pose estimation and refinement(Iterative PnP for speedup) for a planar board and a 5-sided cube that can be used to cover an end effector and track it robustly even when not all faces are visible. We do this using the new Aruco implementation.

The main novelty is using an Aruco gridboard (planar) layout to createa 3x3 pattern which could be each face of a 5-sided cube after the 4 corner faces are cut out. In the detection phase, I used a board pattern(3-D) and located each marker corner with respect to the world frame
located at the center of the bottom face. This creates a robust target that can be located even when only one face is visible or multiple faces are partially visible.
