# Image merger

from email.mime import image

import cv2
import numpy as np
from colorama import Fore, Style, init

file1 = "../LiDART_Script/images/30mW/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE0.png"
file2 = "../LiDART_Script/images/30mW/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE1.png"
file3 = "../LiDART_Script/images/30mW/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE2.png"
file4 = "../LiDART_Script/images/30mW/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE3.png"
file5 = "../LiDART_Script/images/30mW/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE4.png"
file6 = "../LiDART_Script/images/30mW/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE5.png"
file7 = "../LiDART_Script/images/30mW/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE6.png"
file8 = "../LiDART_Script/images/30mW/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE7.png"

image1 = cv2.imread(file1, cv2.IMREAD_ANYDEPTH)
image2 = cv2.imread(file2, cv2.IMREAD_ANYDEPTH)
image3 = cv2.imread(file3, cv2.IMREAD_ANYDEPTH)
image4 = cv2.imread(file4, cv2.IMREAD_ANYDEPTH)
image5 = cv2.imread(file5, cv2.IMREAD_ANYDEPTH)
image6 = cv2.imread(file6, cv2.IMREAD_ANYDEPTH)
image7 = cv2.imread(file7, cv2.IMREAD_ANYDEPTH)
image8 = cv2.imread(file8, cv2.IMREAD_ANYDEPTH)

h = image1.shape[0]
w = image1.shape[1]

mergedImage = np.zeros((h, w), np.uint16)

mergedImage = cv2.addWeighted(image1, 1/8, mergedImage, 1, 0)
mergedImage = cv2.addWeighted(image2, 1/8, mergedImage, 1, 0)
mergedImage = cv2.addWeighted(image3, 1/8, mergedImage, 1, 0)
mergedImage = cv2.addWeighted(image4, 1/8, mergedImage, 1, 0)
mergedImage = cv2.addWeighted(image5, 1/8, mergedImage, 1, 0)
mergedImage = cv2.addWeighted(image6, 1/8, mergedImage, 1, 0)
mergedImage = cv2.addWeighted(image7, 1/8, mergedImage, 1, 0)
mergedImage = cv2.addWeighted(image8, 1/8, mergedImage, 1, 0)

cv2.imwrite("../LiDART_Script/images/30mW/8m_30mW_50us_GB_OFF_Pulse_BASELINE_MERGED_V2.png", mergedImage)
