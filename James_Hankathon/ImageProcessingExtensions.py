import cv2 as cv
import numpy as np
from PyQt5 import QtCore, QtWidgets
import sys

spaceImg = "assets/Earth.jpg"

# 1
# 1-1
sun = cv.imread(spaceImg)
height_s, width_s = sun.shape[:2]
# 1-2
b, g, r = cv.split(sun)
zeros = np.zeros(sun.shape[:2], dtype="uint8")
b = cv.merge([b, zeros, zeros])
g = cv.merge([zeros, g, zeros])
r = cv.merge([zeros, zeros, r])
# 1-3
B, G, R = cv.split(sun)
sun_g = cv.cvtColor(sun, cv.COLOR_BGR2GRAY)
new_s = sun.copy()
row_s, col_s = sun.shape[0:2]
for i in range(row_s):
    for j in range(col_s):
        new_s[i, j] = sum(sun[i, j]) / 3
# 1-4
strong_d = cv.imread("assets/style4_pic.jpg")
weak_d = cv.imread("assets/style5_pic.jpg")
blend = cv.addWeighted(strong_d, 1, weak_d, 0, 0.0)

# 2
# 2-1
l_WN = cv.imread("assets/style1_pic.jpg")
wn_gb = cv.GaussianBlur(l_WN, (5, 5), 0)
# 2-2
wn_bf = cv.bilateralFilter(l_WN, 9, 90, 90)
# 2-3
l_PS = cv.imread("assets/style2_pic.jpg")
ps_mb3 = cv.medianBlur(l_PS, 3)
ps_mb5 = cv.medianBlur(l_PS, 5)


# 3
def convolution(image, Filter):
    image_rows, image_cols = image.shape
    f_row, f_col = 3, 3
    result = np.zeros((image_rows, image_cols))
    pr = int((f_row - 1) / 2)
    pc = int((f_col - 1) / 2)
    padded = np.zeros((2 + image_rows, 2 + image_cols))
    padded[pr:padded.shape[0] - pr, pc:padded.shape[1] - pc] = image
    for row in range(image_rows):
        for col in range(image_cols):
            result[row, col] = abs(np.sum(Filter * padded[row: row + f_row, col: col + f_col]))
    return result


# 3-1
house = cv.imread("assets/texture1_pic.jpg")
h_h, h_w = house.shape[:2]
house_g = cv.cvtColor(house, cv.COLOR_BGR2GRAY)
x_gb = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
y_gb = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
gaussian_f = np.exp(-(x_gb ** 2 + y_gb ** 2))
gaussian_f *= 1 / gaussian_f.sum()
house_gb = convolution(house_g, gaussian_f)


# 3-2
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
house_sx = convolution(house_gb, sobel_x)
# house_x = house_sx * 255 / house_sx.max(initial=0)

# 3-3
sobel_y = sobel_x.transpose()
house_sy = convolution(house_gb, sobel_y)
# house_y = house_sy * 255 / house_sy.max(initial=0)

# 3-4
house_m = np.sqrt(np.square(house_sx) + np.square(house_sy))
# house_m = house_m * 255 / house_m.max()


# 4
# 4-1
square = cv.imread(spaceImg)
square_r = cv.resize(square, (256, 256))
# 4-2
tx, ty = 0, 60
M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
square_t = cv.warpAffine(src=square_r, M=M, dsize=(400, 300))
# 4-3
width_sq, height_sq = square_t.shape[0:2]
center = (width_sq / 2, height_sq / 2)
R = cv.getRotationMatrix2D(center=center, angle=10, scale=0.5)
square_rs = cv.warpAffine(src=square_t, M=R, dsize=(400, 300))
# 4-4
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
S = cv.getAffineTransform(pts1, pts2)
square_sh = cv.warpAffine(src=square_rs, M=S, dsize=(400, 300))


# 1-1 gui
def Load_Image():
    global sun, height_s, width_s
    cv.imshow("Display window", sun)
    print("Height: " + str(height_s))
    print("Width: " + str(width_s))
    cv.waitKey(0)
    cv.destroyAllWindows()


# 1-2 gui
def Color_Separation():
    global b, g, r
    cv.imshow("B Channel", b)
    cv.imshow("G Channel", g)
    cv.imshow("R Channel", r)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 1-3 gui
def Color_Transformation():
    global sun_g, new_s
    cv.imshow("OpenCV Function: ", sun_g)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 1-4 gui
def Blending():
    global blend
    cv.namedWindow('Blending')
    cv.createTrackbar('Blend', 'Blending', 0, 255, Update)
    cv.imshow('Blending', blend)
    cv.waitKey(0)
    cv.destroyAllWindows()


def Update(val):
    global blend
    alpha = val / 255
    beta = 1 - alpha
    blend = cv.addWeighted(strong_d, beta, weak_d, alpha, 0.0)
    cv.imshow('Blending', blend)


# 2-1 gui
def Gaussian_Blur_2_1():
    global l_WN, wn_gb
    cv.imshow('Original', l_WN)
    cv.imshow('Gaussian Blur', wn_gb)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 2-2 gui
def Bilateral_Filter():
    global l_WN, wn_bf
    cv.imshow('Original', l_WN)
    cv.imshow('Bilateral Filter', wn_bf)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 2-3 gui
def Median_Filter():
    global l_PS, ps_mb3, ps_mb5
    cv.imshow('Original', l_PS)
    cv.imshow('Median Filter 3X3', ps_mb3)
    cv.imshow('Bilateral Filter 5X5', ps_mb5)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 3-1
def Gaussian_Blur_3():
    global house, house_g, house_gb
    cv.imwrite(spaceImg, house_gb)
    house_gb = cv.imread(spaceImg)
    cv.imshow('Original', house)
    cv.imshow('Grayscale', house_g)
    cv.imshow('Gaussian Blur', house_gb)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 3-2
def Sobel_X():
    global house_g, house_sx
    cv.imwrite(spaceImg, house_sx)
    house_sx = cv.imread(spaceImg)
    cv.imshow('Grayscale', house_g)
    cv.imshow('Sobel X', house_sx)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 3-3
def Sobel_Y():
    global house_g, house_sy
    cv.imwrite(spaceImg, house_sy)
    house_sy = cv.imread(spaceImg)
    cv.imshow('Grayscale', house_g)
    cv.imshow('Sobel Y', house_sy)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 3-4
def Magnitude_xy():
    global house_m
    cv.imwrite(spaceImg, house_m)
    house_m = cv.imread(spaceImg)
    cv.imshow('Magnitude', house_m)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 4-1 gui
def Resize_():
    global square, square_r
    # cv.imshow("Original", square)
    cv.imshow("Resize: ", square_r)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 4-2 gui
def Translate_():
    global square_t
    cv.imshow("Translation:", square_t)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 4-3 gui
def Rotate_Scale_():
    global square_rs
    cv.namedWindow("Rotate and Scale: ", cv.WINDOW_NORMAL)
    cv.resizeWindow("Rotate and Scale: ", 400, 300)
    cv.imshow("Rotate and Scale: ", square_rs)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 4-4 gui
def Shear_():
    global square_sh
    cv.imshow("Shear: ", square_sh)
    cv.waitKey(0)
    cv.destroyAllWindows()


class Ui_Dialog(object):
    def __init__(self, dialog):
        self.groupBox = QtWidgets.QGroupBox(dialog)
        self.Q11 = QtWidgets.QPushButton(self.groupBox)
        self.Q12 = QtWidgets.QPushButton(self.groupBox)
        self.Q13 = QtWidgets.QPushButton(self.groupBox)
        self.Q14 = QtWidgets.QPushButton(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(dialog)
        self.Q21 = QtWidgets.QPushButton(self.groupBox_2)
        self.Q22 = QtWidgets.QPushButton(self.groupBox_2)
        self.Q23 = QtWidgets.QPushButton(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(dialog)
        self.Q31 = QtWidgets.QPushButton(self.groupBox_3)
        self.Q32 = QtWidgets.QPushButton(self.groupBox_3)
        self.Q33 = QtWidgets.QPushButton(self.groupBox_3)
        self.Q34 = QtWidgets.QPushButton(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(dialog)
        self.Q41 = QtWidgets.QPushButton(self.groupBox_4)
        self.Q42 = QtWidgets.QPushButton(self.groupBox_4)
        self.Q43 = QtWidgets.QPushButton(self.groupBox_4)
        self.Q44 = QtWidgets.QPushButton(self.groupBox_4)
        self.setupUi(dialog)

    def setupUi(self, dialog):
        dialog.setObjectName("Dialog")
        dialog.resize(700, 400)
        # 1
        self.groupBox.setGeometry(QtCore.QRect(20, 50, 150, 300))
        self.groupBox.setObjectName("groupBox")
        self.Q11.setGeometry(QtCore.QRect(5, 50, 140, 25))
        self.Q11.setObjectName("Q11")
        self.Q11.clicked.connect(Load_Image)
        self.Q12.setGeometry(QtCore.QRect(5, 120, 140, 25))
        self.Q12.setObjectName("Q12")
        self.Q12.clicked.connect(Color_Separation)
        self.Q13.setGeometry(QtCore.QRect(5, 190, 140, 25))
        self.Q13.setObjectName("Q13")
        self.Q13.clicked.connect(Color_Transformation)
        self.Q14.setGeometry(QtCore.QRect(5, 250, 140, 25))
        self.Q14.setObjectName("Q14")
        self.Q14.clicked.connect(Blending)
        # 2
        self.groupBox_2.setGeometry(QtCore.QRect(190, 50, 150, 300))
        self.groupBox_2.setObjectName("groupBox_2")
        self.Q21.setGeometry(QtCore.QRect(5, 70, 140, 25))
        self.Q21.setObjectName("Q21")
        self.Q21.clicked.connect(Gaussian_Blur_2_1)
        self.Q22.setGeometry(QtCore.QRect(5, 140, 140, 25))
        self.Q22.setObjectName("Q22")
        self.Q22.clicked.connect(Bilateral_Filter)
        self.Q23.setGeometry(QtCore.QRect(5, 210, 140, 25))
        self.Q23.setObjectName("Q23")
        self.Q23.clicked.connect(Median_Filter)
        # 3
        self.groupBox_3.setGeometry(QtCore.QRect(360, 50, 150, 300))
        self.groupBox_3.setObjectName("groupBox_3")
        self.Q31.setGeometry(QtCore.QRect(5, 50, 140, 25))
        self.Q31.setObjectName("Q31")
        self.Q31.clicked.connect(Gaussian_Blur_3)
        self.Q32.setGeometry(QtCore.QRect(5, 120, 140, 25))
        self.Q32.setObjectName("Q32")
        self.Q32.clicked.connect(Sobel_X)
        self.Q33.setGeometry(QtCore.QRect(5, 190, 140, 25))
        self.Q33.setObjectName("Q33")
        self.Q33.clicked.connect(Sobel_Y)
        self.Q34.setGeometry(QtCore.QRect(5, 250, 140, 25))
        self.Q34.setObjectName("Q34")
        self.Q34.clicked.connect(Magnitude_xy)
        # 4
        self.groupBox_4.setGeometry(QtCore.QRect(530, 50, 150, 300))
        self.groupBox_4.setObjectName("groupBox_4")
        self.Q41.setGeometry(QtCore.QRect(5, 50, 140, 25))
        self.Q41.setObjectName("Q41")
        self.Q41.clicked.connect(Resize_)
        self.Q42.setGeometry(QtCore.QRect(5, 120, 140, 25))
        self.Q42.setObjectName("Q42")
        self.Q42.clicked.connect(Translate_)
        self.Q43.setGeometry(QtCore.QRect(5, 190, 140, 25))
        self.Q43.setObjectName("Q43")
        self.Q43.clicked.connect(Rotate_Scale_)
        self.Q44.setGeometry(QtCore.QRect(5, 250, 140, 25))
        self.Q44.setObjectName("Q44")
        self.Q44.clicked.connect(Shear_)

        self.groupBox.setTitle("1.Image Processing")
        self.Q11.setText("1.1 Load Image")
        self.Q12.setText("1.2 Color Separation")
        self.Q13.setText("1.3 Grayscale")
        self.Q14.setText("1.4 Blending")
        self.groupBox_2.setTitle("2. Image Smoothing")
        self.Q21.setText("2.1 Gaussian Blur")
        self.Q22.setText("2.2 Bilateral Filter")
        self.Q23.setText("2.3 Median Filter")
        self.groupBox_3.setTitle("3. Edge Detection")
        self.Q31.setText("3.1 Gaussian Blur")
        self.Q32.setText("3.2 Sobel X")
        self.Q33.setText("3.3 Sobel Y")
        self.Q34.setText("3.4 Magnitude")
        self.groupBox_4.setTitle("4. Transformation")
        self.Q41.setText("4.1 Resize")
        self.Q42.setText("4.2 Translation")
        self.Q43.setText("4.3 Rotation, Scaling")
        self.Q44.setText("4.4 Shearing")
        QtCore.QMetaObject.connectSlotsByName(dialog)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
