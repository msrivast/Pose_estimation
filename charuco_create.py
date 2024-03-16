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


if __name__ == "__main__":
    createBoard()