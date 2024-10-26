import cv2
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style, init

img = cv2.imread(".\\LiDART_Script\\images\\8bitImage.png", cv2.IMREAD_UNCHANGED)
img24 = cv2.imread(".\\LiDART_Script\\images\\12bitImage.png", cv2.IMREAD_UNCHANGED)

# Check if the image has been opened correctly
if img is None:
    print(Fore.RED + "Error opening the image")


#cv2.imshow("Image", img)
#cv2.waitKey(0)    

# Crop the bottom of the image
#img = img[0:img.shape[0] - 100, 0:img.shape[1]]

# Find the max pixel coordinates
maxPixelCoordinates8 = np.where(img == np.amax(img))
maxPixelCoordinates24 = np.where(img24 == np.amax(img24))

# Get the max pixel value
maxPixelValue8 = np.amax(img)
maxPixelValue24 = np.amax(img24)

print(Fore.YELLOW + "Max pixel value: " + str(maxPixelValue8))
print(Fore.YELLOW + "Max pixel value: " + str(maxPixelValue24))

# Do an histogram of all the pixel values (16 bit)
plt.hist(img.ravel(), 512, [0, maxPixelValue8])
plt.ylim(0, 150)
plt.show()

# Do an histogram of all the pixel values (16 bit)
plt.hist(img.ravel(), 512, [0, maxPixelValue24])
plt.ylim(0, 150)
plt.show()