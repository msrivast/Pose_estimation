# 08/18/23 - Manu Srivastava
# Use OpenCV 4.8 to do pose estimation for a 3D Aruco board
# Camera I have currently(microsoft Lifecam HD-3000) probably has a focal length of ~1m
# start with the assumption that intrinsic calibration and distortion params are known
 
import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# width = 10
# height = 8
# board_size = (width, height)
square_length = 0.072
marker_length = 0.056

right_face_points = np.array([[square_length/2,  marker_length/2,   marker_length/2 - square_length/2],
                              [square_length/2,  marker_length/2, - marker_length/2 - square_length/2],
                              [square_length/2, -marker_length/2, - marker_length/2 - square_length/2],
                              [square_length/2, -marker_length/2,   marker_length/2 - square_length/2]])

left_face_points = np.array([[-square_length/2,  marker_length/2, - marker_length/2 - square_length/2],
                             [-square_length/2,  marker_length/2,   marker_length/2 - square_length/2],
                             [-square_length/2, -marker_length/2,   marker_length/2 - square_length/2],
                             [-square_length/2, -marker_length/2, - marker_length/2 - square_length/2]])

back_face_points = np.array([[-marker_length/2, -square_length/2,   marker_length/2 - square_length/2],
                             [ marker_length/2, -square_length/2,   marker_length/2 - square_length/2],
                             [ marker_length/2, -square_length/2, - marker_length/2 - square_length/2],
                             [-marker_length/2, -square_length/2, - marker_length/2 - square_length/2]])

front_face_points = np.array([[-marker_length/2, square_length/2, - marker_length/2 - square_length/2],
                              [ marker_length/2, square_length/2, - marker_length/2 - square_length/2],
                              [ marker_length/2, square_length/2,   marker_length/2 - square_length/2],
                              [-marker_length/2, square_length/2,   marker_length/2 - square_length/2]])

object_points = np.array([np.array([[-0.5,0.5,0.0],[0.5,0.5,0.0],[0.5,-0.5,0.0],[-0.5,-0.5,0.0]])*marker_length,right_face_points, left_face_points,back_face_points,front_face_points]) #Only the bottom face points are clockwise

board_marker_ids = np.array([0,1,2,4,3])
board = cv2.aruco.Board(object_points.astype(np.float32), aruco_dict, board_marker_ids)

detector_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# Calibration params 
camera_matrix = np.array([[1.11512980e+03, 0.00000000e+00, 5.75994080e+02],
                          [0.00000000e+00, 1.10715222e+03, 3.40489981e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeff = np.array([ 2.47900461e-01, -1.68800873e+00, -2.35525689e-03, -7.11035017e-03, 3.07968349e+00])


def draw_axis(frame, camera_matrix, dist_coeff, board, verbose=True):
    frame = cv2.undistort(frame, camera_matrix,dist_coeff)
    marker_corners, marker_ids, rejected_points = aruco_detector.detectMarkers(frame)
    
    if not (marker_ids is None) and len(marker_ids) > 0:
        try:
            obj_points, img_points = board.matchImagePoints(marker_corners, marker_ids)
            flag, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeff)
            if flag:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeff, rvec, tvec, .1)
                cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
                # cv2.aruco.drawDetectedMarkers(frame, rejected_points, borderColor=(100, 0, 240))
                if verbose:
                    print('Translation : {0}'.format(tvec))
                    print('Rotation    : {0}'.format(rvec))
                    print('Distance from camera: {0} m'.format(np.linalg.norm(tvec)))
        except cv2.error as error_inst:
            print("SolvePnP recognizes calibration pattern as non-planar pattern. To process this need to use "
                    "minimum 6 points. The planar pattern may be mistaken for non-planar if the pattern is "
                    "deformed or incorrect camera parameters are used.")
            print(error_inst.err)
            return None

    return frame

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #Camera resolution MS Lifecam HD-3000
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cap.read()
        k = cv2.waitKey(1)
        if k == 27:  # Esc
            break
        axis_frame = draw_axis(frame, camera_matrix, dist_coeff, board, True)
        cv2.imshow('Out', axis_frame)



if __name__ == '__main__':
    main()