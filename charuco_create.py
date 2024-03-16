import cv2
from cv2 import aruco

def createBoard():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # board = aruco.CharucoBoard((10,8), 0.015,0.011,dictionary)
    board = aruco.CharucoBoard((10,8), 100,70,dictionary)
    board_image = board.generateImage((1020,820), 10, 1)
    cv2.imshow("board", board_image)
    key = cv2.waitKey(1000)
    cv2.imwrite("charuco_board.png", board_image)

def createGridBoard():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # board = aruco.CharucoBoard((10,8), 0.015,0.011,dictionary)
    square_length = 0.07
    marker_length = 0.06
    board = cv2.aruco.GridBoard((3,3), marker_length, square_length - marker_length, dictionary)
    # board_image = board.generateImage((3*square_length*1000,3*square_length*1000),0,1)
    board_image = board.generateImage((700,700),60,1)
    cv2.imshow("board", board_image)
    key = cv2.waitKey(1000)
    cv2.imwrite("aruco_board_3D.png", board_image)


if __name__ == "__main__":
    createGridBoard()