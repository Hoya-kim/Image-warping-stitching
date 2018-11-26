import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy.linalg as lin

from skimage.transform import warp

# Image read
f1 = plt.imread('./images/nachtwacht1.jpg') / 255
f2 = plt.imread('./images/nachtwacht2.jpg') / 255

# Setting matching points in first image
xy_1 = np.array([[157, 32],  # x1[0][0], y1[0][1]
                 [211, 37],  # x2[1][0], y2[1][1]
                 [222, 107],  # x3[2][0], y3[2][1]
                 [147, 124]])  # x4[3][0], y4[3][1]

# Setting matching points in second image
xy_2 = np.array([[6, 38],  # x'1[0][0], y'1[0][1]
                 [56, 31],  # x'2[1][0], y'2[1][1]
                 [82, 85],  # x'3[2][0], y'3[2][1]
                 [22, 118]])  # x'4[3][0], y'4[3][1]

# Get homography matrix with cv2 module
# cv2_Homography = cv2.getPerspectiveTransform(xy_1.astype(np.float32), xy_2.astype(np.float32))

# Make homography matrix
# Make matrix A
arrayA = np.array([[xy_1[0][0], xy_1[0][1], 1, 0, 0, 0, -xy_1[0][0] * xy_2[0][0], -xy_1[0][1] * xy_2[0][0]],
                   [0, 0, 0, xy_1[0][0], xy_1[0][1], 1, -xy_1[0][0] * xy_2[0][1], -xy_1[0][1] * xy_2[0][1]],
                   [xy_1[1][0], xy_1[1][1], 1, 0, 0, 0, -xy_1[1][0] * xy_2[1][0], -xy_1[1][1] * xy_2[1][0]],
                   [0, 0, 0, xy_1[1][0], xy_1[1][1], 1, -xy_1[1][0] * xy_2[1][1], -xy_1[1][1] * xy_2[1][1]],
                   [xy_1[2][0], xy_1[2][1], 1, 0, 0, 0, -xy_1[2][0] * xy_2[2][0], -xy_1[2][1] * xy_2[2][0]],
                   [0, 0, 0, xy_1[2][0], xy_1[2][1], 1, -xy_1[2][0] * xy_2[2][1], -xy_1[2][1] * xy_2[2][1]],
                   [xy_1[3][0], xy_1[3][1], 1, 0, 0, 0, -xy_1[3][0] * xy_2[3][0], -xy_1[3][1] * xy_2[3][0]],
                   [0, 0, 0, xy_1[3][0], xy_1[3][1], 1, -xy_1[3][0] * xy_2[3][1], -xy_1[3][1] * xy_2[3][1]]])
M_A = np.asmatrix(arrayA)

# transpose matrix A
M_At = M_A.T

# Make matrix b
arrayB = np.array([[xy_2[0][0]],
                   [xy_2[0][1]],
                   [xy_2[1][0]],
                   [xy_2[1][1]],
                   [xy_2[2][0]],
                   [xy_2[2][1]],
                   [xy_2[3][0]],
                   [xy_2[3][1]]])
M_B = np.asmatrix(arrayB)

# Make homography matrix
My_Homography = np.asarray((M_At * M_A).I * (M_At * M_B))

My_Homography_arr = np.array([[My_Homography[0][0], My_Homography[1][0], My_Homography[2][0]],
                              [My_Homography[3][0], My_Homography[4][0], My_Homography[5][0]],
                              [My_Homography[6][0], My_Homography[7][0], 1]])

# My_homography's inverse
inv_my_homo = lin.inv(My_Homography_arr)

# Homographic transformation with cv2 module
# cv2_warped = warp(f2, cv2_Homography, output_shape=(300, 550))

# Make blank image
warped = np.zeros((300, 550, 3), np.float32)

# Make warping source
for y in range(0, 300):
    for x in range(0, 550):
        pixel = np.array([[x],
                          [y],
                          [1]])

        # Resource_pixel = np.asarray(np.asmatrix(inv_my_homo) * np.asmatrix(pixel))
        Resource_pixel = np.asarray(np.asmatrix(My_Homography_arr) * np.asmatrix(pixel))

        # Coordinate scale
        trans_x = Resource_pixel[0][0] / Resource_pixel[2][0]
        trans_y = Resource_pixel[1][0] / Resource_pixel[2][0]

        # Continue that pixels not matched
        if (trans_x < 0 or trans_y < 0
                or trans_x > f2.shape[1] - 1 or trans_y > f2.shape[0] - 1):
            continue

        # divided by two parts with integer & decimal number
        tx = int(trans_x)
        ty = int(trans_y)
        a = trans_x - tx
        b = trans_y - ty

        # Bilinear Interpolation
        warped[y][x] = ((((1.0 - a) * (1.0 - b)) * f2[ty][tx])
                        + ((a * (1.0 - b)) * f2[ty][tx + 1])
                        + ((a * b) * f2[ty + 1][tx + 1])
                        + (((1.0 - a) * b) * f2[ty + 1][tx]))

# Image stitch
M, N = f1.shape[:2]
f_stitched = np.copy(warped)
f_stitched[0:M, 0:N, :] = f1

# Plot
plt.subplot(221)
plt.imshow(f1)
plt.axis('off')
plt.scatter(xy_1[:, 0], xy_1[:, 1], marker='x')
plt.subplot(222)
plt.imshow(f2)
plt.axis('off')
plt.scatter(xy_2[:, 0], xy_2[:, 1], marker='x')
plt.subplot(223)
plt.imshow(warped)
plt.axis('off')
plt.subplot(224)
plt.imshow(f_stitched)
plt.axis('off')
plt.show()
