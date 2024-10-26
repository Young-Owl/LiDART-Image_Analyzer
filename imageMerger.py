import json
import os
import pprint as pp
from re import Pattern

import cv2
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style, init

from imageAnalyzer import (
    EXPOSURE,
    FILE_PATTERNS,
    KEYWORD_FOCUS,
    KEYWORD_GB,
    POTENCY_VALUES,
)

IMAGES_PATH = "../LiDART_Script/images/"
POTENCY     = "30mW"
EXPOSURE    = "50us"
NUM_PICS    = np.uint8(8)
DISTANCES   = ["1m", "2m", "4m", "6m", "8m"]

def main():
    init(autoreset=True)
    patternFiles = POTENCY_VALUES
    subFolderPath = [IMAGES_PATH + POTENCY]

    for subFolder in subFolderPath:
            files = os.listdir(subFolder)
            for file in files:
                for pattern, value in FILE_PATTERNS.items():
                    if pattern in file:
                        for keyword in KEYWORD_GB:
                            if keyword in file:
                                for keyword2 in KEYWORD_FOCUS:
                                    if keyword2 in file:
                                        patternFiles[POTENCY][value][keyword][keyword2].append(file)
                        break
                    
    #print(Fore.YELLOW + "Files found for 1mW,50,Off,1: ")
    #print(patternFiles["1mW"]["50us"]["GB_OFF"]["F1"])

    # According to the POTENCY value, we will merge the images of the said potency
    # We will merge the images of the same exposure,focus and GB value

    for gb in patternFiles[POTENCY][EXPOSURE]:
        for focus in patternFiles[POTENCY][EXPOSURE][gb]:

            images1m = []; images2m = []; images4m = []; images6m = []; images8m = []
            mergedImage1m = 0; mergedImage2m = 0; mergedImage4m = 0; mergedImage6m = 0; mergedImage8m = 0
            print(Fore.GREEN + "Merging images for " + POTENCY + " " + EXPOSURE + " " + gb + " " + focus)
            
            for file in patternFiles[POTENCY][EXPOSURE][gb][focus]:
                if str(DISTANCES[0]) in file:
                    images1m.append(cv2.imread(subFolderPath[0] + "/" + file, cv2.IMREAD_ANYDEPTH))
                    print("1m   " + file)
                elif str(DISTANCES[1]) in file:
                    images2m.append(cv2.imread(subFolderPath[0] + "/" + file, cv2.IMREAD_ANYDEPTH))
                    print("2m   " + file)
                elif str(DISTANCES[2]) in file:
                    images4m.append(cv2.imread(subFolderPath[0] + "/" + file, cv2.IMREAD_ANYDEPTH))
                    print("4m   " + file)
                elif str(DISTANCES[3]) in file:
                    images6m.append(cv2.imread(subFolderPath[0] + "/" + file, cv2.IMREAD_ANYDEPTH))
                    print("6m   " + file)
                elif str(DISTANCES[4]) in file:
                    images8m.append(cv2.imread(subFolderPath[0] + "/" + file, cv2.IMREAD_ANYDEPTH))
                    print("8m   " + file)
                    
            for i in range(NUM_PICS):
                mergedImage1m += images1m[i]/NUM_PICS
                mergedImage2m += images2m[i]/NUM_PICS
                mergedImage4m += images4m[i]/NUM_PICS
                mergedImage6m += images6m[i]/NUM_PICS
                mergedImage8m += images8m[i]/NUM_PICS
            
            mergedImage1m = np.uint16(mergedImage1m)
            mergedImage2m = np.uint16(mergedImage2m)
            mergedImage4m = np.uint16(mergedImage4m)
            mergedImage6m = np.uint16(mergedImage6m)
            mergedImage8m = np.uint16(mergedImage8m)
            
            #print(Fore.GREEN + "Images merged for " + powerLevel + " " + exposure + " " + gb + " " + focus)
            cv2.imwrite(IMAGES_PATH[0] + "/" + POTENCY + "_1m_" + EXPOSURE + "_" + gb + "_" + focus + "_MERGED_AVG.png", mergedImage1m)
            cv2.imwrite(IMAGES_PATH[0] + "/" + POTENCY + "_2m_" + EXPOSURE + "_" + gb + "_" + focus + "_MERGED_AVG.png", mergedImage2m)
            cv2.imwrite(IMAGES_PATH[0] + "/" + POTENCY + "_4m_" + EXPOSURE + "_" + gb + "_" + focus + "_MERGED_AVG.png", mergedImage4m)
            cv2.imwrite(IMAGES_PATH[0] + "/" + POTENCY + "_6m_" + EXPOSURE + "_" + gb + "_" + focus + "_MERGED_AVG.png", mergedImage6m)
            cv2.imwrite(IMAGES_PATH[0] + "/" + POTENCY + "_8m_" + EXPOSURE + "_" + gb + "_" + focus + "_MERGED_AVG.png", mergedImage8m)
            
            print(Fore.GREEN + "Images saved for " + POTENCY + " " + EXPOSURE + " " + gb + " " + focus)
            print(Style.RESET_ALL)
    
    print(Fore.GREEN + "All images merged and saved!")
    
if __name__ == "__main__":
    main()

