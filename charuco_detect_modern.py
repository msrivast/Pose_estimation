# 08/17/23 - Manu Srivastava
# Use OpenCV 4.8 or newer to do pose estimation and refinement for charuco board
# I am using Charuco becuase I used it for calibration and had it handy. After calibration, Charuco
# provides no advantage over any other board for the Perspective-n-Point problem. 
# Camera I have currently(microsoft Lifecam HD-3000) probably has a focal length of ~1m
# start with the assumption that intrinsic calibration and distortion params are known
 

import cv2
import numpy as np
# from cv2 import aruco

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
width = 10
height = 8
board_size = (width, height)
square_len = 0.0161423 #m
marker_len = 0.7*square_len
board = cv2.aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)

charuco_detector = cv2.aruco.CharucoDetector(board)

# Calibration params 
camera_matrix = np.array([[1.11512980e+03,0.00000000e+00, 5.75994080e+02],
                          [0.00000000e+00, 1.10715222e+03, 3.40489981e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeff = np.array([ 2.47900461e-01, -1.68800873e+00, -2.35525689e-03, -7.11035017e-03, 3.07968349e+00])
# camera_matrix = np.array([[0.001, 0.0, 0.0],
#                           [0.0, 0.001, 0.0],
#                           [0.0, 0.0, 1.0]])
# dist_coeff = np.array([ 0.0, 0.0, 0.0, 0.0, 0.0]) #Doesn't do too badly, even if BS values are provided!

first_run = True
def draw_axis(frame, camera_matrix, dist_coeff, board, verbose=True):
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame)
    
    if not (charuco_ids is None) and len(charuco_ids) >= 4:
        try:
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
            global rvec, tvec, first_run
            if first_run:
                flag, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeff)
                first_run = False

            flag, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeff,rvec,tvec,useExtrinsicGuess=True,flags=0)

            if flag:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeff, rvec, tvec, .1)
                if verbose:
                    print('Translation : {0}'.format(tvec))
                    print('Rotation    : {0}'.format(rvec))
                    print('Distance from camera: {0} m'.format(np.linalg.norm(tvec)))
        except cv2.error as error_inst:
            # print("SolvePnP recognizes calibration pattern as non-planar pattern. To process this need to use "
            #         "minimum 6 points. The planar pattern may be mistaken for non-planar if the pattern is "
            #         "deformed or incorrect camera parameters are used.")
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
        if axis_frame is not None:
            cv2.imshow('Java', axis_frame)
        else:
            cv2.imshow('Java', frame)
            # cv2.imshow('Java_no_detect', frame)


if __name__ == '__main__':
    main()