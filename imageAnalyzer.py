import copy
import dis
import imghdr
import json
import os
import pprint as pp
import re
import tkinter as tk
from array import array
from audioop import avg
from email.mime import base, image
from math import pi
from shlex import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style, init
from natsort import natsorted, ns, os_sorted

# -------- Global DEFINES --------
# These defines are set manually to choose which set of pics will be showm, analyzed and plotted
IMAGES_PATH = "../LiDART_Script/images/good/25jul/"   # Path to the images
POTENCY     = "30mW"                                         # Potency of the laser 
EXPOSURE    = "50us"                                        # Exposure time
GB          = "GB_ON"                                       # Gain boost
FOCUS       = "F1"                                        # Focus 

#######################




########################
# Initialize the dictionary that will store the results
POTENCY_VALUES = {
    "1mW": {
        "50us": {
            "GB_OFF": {"F1": [], "Finf": []},
            "GB_ON": {"F1": [], "Finf": []}
        },
        "100us": {
            "GB_OFF": {"F1": [], "Finf": []},
            "GB_ON": {"F1": [], "Finf": []}
        }
    },
    "10mW": {
        "50us": {
            "GB_OFF": {"F1": [], "Finf": []},
            "GB_ON": {"F1": [], "Finf": []}
        },
        "100us": {
            "GB_OFF": {"F1": [], "Finf": []},
            "GB_ON": {"F1": [], "Finf": []}
        }
    },
    "20mW": {
        "50us": {
            "GB_OFF": {"F1": [], "Finf": []},
            "GB_ON": {"F1": [], "Finf": []}
        },
        "100us": {
            "GB_OFF": {"F1": [], "Finf": []},
            "GB_ON": {"F1": [], "Finf": []}
        }
    },
    "30mW": {
        "50us": {
            "GB_OFF": {"F1": [], "Finf": []},
            "GB_ON": {"F1": [], "Finf": []}
        },
        "100us": {
            "GB_OFF": {"F1": [], "Finf": []},
            "GB_ON": {"F1": [], "Finf": []}
        }
    }
}

# Patterns for the file names; These are to identify the file names
FILE_PATTERNS   = {
    "5e-05": "50us",
    "0.0001": "100us",
}
KEYWORD_GB     = ["GB_OFF", "GB_ON"]
KEYWORD_FOCUS   = ["F1", "Finf"]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img = param
        print("Pixel value at (", x, ",", y, "):", img[y, x])

def main():
    
    # Tkinter initialization, merely for getting the screen resolution
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Colorama initialization
    init(autoreset=True)
    print(Fore.BLUE + "---- Image Analyzer ----")
   
    # Initialize the dictionaries that will store the results
    result = copy.deepcopy(POTENCY_VALUES)   # Create a deep copy of the potencyValues dictionary
    resultMax = copy.deepcopy(POTENCY_VALUES)   # Create a deep copy of the potencyValues dictionary
    resultsPotency = copy.deepcopy(POTENCY_VALUES)   # Create a deep copy of the potencyValues dictionary
    data = copy.deepcopy(POTENCY_VALUES)     # Create a deep copy of the potencyValues dictionary
    dataMax = copy.deepcopy(POTENCY_VALUES)     # Create a deep copy of the potencyValues dictionary
    dataPotency = copy.deepcopy(POTENCY_VALUES)     # Create a deep copy of the potencyValues dictionary
    
    # List all files that have specific keywords in an arraynand then print them
    #subFolderPath = [IMAGES_PATH + POTENCY]
    subFolderPath = [IMAGES_PATH + "30mW/"]

    # Loop through the subfolders and find the files that match the patterns given by the defines
    for subFolder in subFolderPath:
        files = os.listdir(subFolder)
        for file in files:
            for pattern, value in FILE_PATTERNS.items():
                if pattern in file:
                    for keyword in KEYWORD_GB:
                        if keyword in file:
                            for keyword2 in KEYWORD_FOCUS:
                                if keyword2 in file:
                                    POTENCY_VALUES[POTENCY][value][keyword][keyword2].append(file)
                    break

    
    
    # Sort the files in the dictionary from the smallest to the largest distance with os_sorted
    for potency in POTENCY_VALUES:
        for exposure in POTENCY_VALUES[potency]:
            for gb in POTENCY_VALUES[potency][exposure]:
                for focus in POTENCY_VALUES[potency][exposure][gb]:
                    POTENCY_VALUES[potency][exposure][gb][focus] = os_sorted(POTENCY_VALUES[potency][exposure][gb][focus])

    print(Fore.YELLOW + "Files found for 30mW,50,Off,1: ")
    print(POTENCY_VALUES["30mW"]["50us"]["GB_OFF"]["F1"])
    
    """ 
    # Print the files found
    for potency in POTENCY_VALUES:
        for exposure in POTENCY_VALUES[potency]:
            for gb in POTENCY_VALUES[potency][exposure]:
                for focus in POTENCY_VALUES[potency][exposure][gb]:
                    print(Fore.YELLOW + "Files found for %s,%s,%s,%s: " % (potency, exposure, gb, focus))
                    print(POTENCY_VALUES[potency][exposure][gb][focus])
    
    """
    
    # User input
    treshold = int(80)
    distances = [1,2,4,8,16,24,46]
    squareSize = 200
    squareSizeBackup = squareSize
    squareDecrease = 25
    
    pixels = np.zeros(len(distances))
    maxVal = np.zeros(len(distances))
    potencyCnt = np.zeros(len(distances))
    
    k = 0

    # Read the baseline images (Alter the path to the baseline images if necessary)
    baselineFiles = {   "./baselinePics/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE0.png",
                        "./baselinePics/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE1.png",
                        "./baselinePics/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE2.png",
                        "./baselinePics/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE3.png",
                        "./baselinePics/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE4.png",
                        "./baselinePics/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE5.png",
                        "./baselinePics/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE6.png",
                        "./baselinePics/30mW_8m_5e-05_GB_OFF_Pulse_BASELINE7.png",
                    }
                     
                     
    baselineFilesGB = { "./baselinePics/30mW_8m_5e-05_GB_ON_Pulse_BASELINE0.png",
                        "./baselinePics/30mW_8m_5e-05_GB_ON_Pulse_BASELINE1.png",
                        "./baselinePics/30mW_8m_5e-05_GB_ON_Pulse_BASELINE2.png",
                        "./baselinePics/30mW_8m_5e-05_GB_ON_Pulse_BASELINE3.png",
                        "./baselinePics/30mW_8m_5e-05_GB_ON_Pulse_BASELINE4.png",
                        "./baselinePics/30mW_8m_5e-05_GB_ON_Pulse_BASELINE5.png",
                        "./baselinePics/30mW_8m_5e-05_GB_ON_Pulse_BASELINE6.png",
                        "./baselinePics/30mW_8m_5e-05_GB_ON_Pulse_BASELINE7.png",
                    }
    
    valsBaseline = []
    stdDev = 0
    avgPixelValue = 0
    
    valsBaselineGB = []
    stdDevGB = 0
    avgPixelValueGB = 0

    # Read the baseline image
    for baselineFile in baselineFiles:
        baseline = cv2.imread(baselineFile, cv2.IMREAD_ANYDEPTH)
        baseline = baseline[0:baseline.shape[0] - 100, 0:baseline.shape[1]]
        stdDev += np.std(baseline)
        avgPixelValue += round(np.average(baseline))
        baseline = baseline.flatten()
        for i in range(len(baseline)):
            valsBaseline.append(baseline[i])

    stdDev = stdDev/len(baselineFiles)
    avgPixelValue = avgPixelValue/len(baselineFiles)
    
    for baselineFile in baselineFilesGB:
        baseline = cv2.imread(baselineFile, cv2.IMREAD_ANYDEPTH)
        baseline = baseline[0:baseline.shape[0] - 100, 0:baseline.shape[1]]
        stdDevGB += np.std(baseline)
        avgPixelValueGB += round(np.average(baseline))
        baseline = baseline.flatten()
        for i in range(len(baseline)):
            valsBaselineGB.append(baseline[i])

    stdDevGB = stdDevGB/len(baselineFilesGB)
    avgPixelValueGB = avgPixelValueGB/len(baselineFilesGB)
    
    print(Fore.BLUE + "Standard deviation: " + str(stdDev))
    print(Fore.BLUE + "Standard deviation GB: " + str(stdDevGB))
    
    avgPixelValue = round(avgPixelValue)
    avgPixelValueGB = round(np.average(avgPixelValueGB))
    print(Fore.BLUE + "Average pixel value: " + str(avgPixelValue))
    print(Fore.BLUE + "Average pixel value GB: " + str(avgPixelValueGB))
    
    # Histogram of the baseline images
     
    """ plt.hist(valsBaseline, bins=256, density=False, alpha=0.6, color='k', edgecolor='black', linewidth=1.1)
    plt.xlabel('Pixel value')
    plt.ylabel('Number of pixels')
    plt.title('Baseline histogram (GB_OFF)')
    plt.xlim(int(avgPixelValue - 100), int(avgPixelValue + 100))
    plt.xticks(np.arange(min(np.unique(valsBaseline)), max(np.unique(valsBaseline))+1, 14))
    plt.xticks(fontsize=6)
    plt.show() """
    
    plt.hist(valsBaselineGB, bins=256, density=False, alpha=0.6, color='k', edgecolor='black', linewidth=1.1)
    plt.xlabel('Pixel value')
    plt.ylabel('Number of pixels')
    plt.title('Baseline histogram (GB_ON)')
    plt.xlim(int(avgPixelValueGB - 100), int(avgPixelValueGB + 100))
    plt.xticks(fontsize=6)
    plt.xticks(np.arange(min(np.unique(valsBaselineGB)), 266+1, 14))
    plt.show()
    

        
    img = {}; img_noBaseline = {}; img_blackNwhite = {}
    img_16bit = {}; img_16bitBase = {}; img_16bitBW = {}
    joinedVert = {}; joinedVert_16bit = {}
    maxPixelCoordinatesArray = {}
    for imageFile in POTENCY_VALUES[POTENCY][EXPOSURE][GB][FOCUS]:
        # Each distance, reduce the square size
        squareSize = squareSize - squareDecrease
        print(Fore.YELLOW + "Square size: " + str(squareSize))
        
        if "24m" in imageFile:
            treshold = 80
        elif"46m" in imageFile:
            treshold = 80
        
        #print(imageFile)
        #img = cv2.imread(readPath + imageFile, cv2.IMREAD_ANYDEPTH)
        
        img[imageFile] = cv2.imread(subFolderPath[0] + imageFile, cv2.IMREAD_ANYDEPTH)
        
        # Check if the image has been opened correctly
        if img[imageFile] is None:
            print(Fore.RED + "Error opening the image")
            break
        
        # Crop the bottom of the image
        img[imageFile] = img[imageFile][0:img[imageFile].shape[0] - 100, 0:img[imageFile].shape[1]]
        img_noBaseline[imageFile] = img[imageFile].copy()
        img_blackNwhite[imageFile] = img[imageFile].copy()
        
        # Temporary - Crop the top of the image
        if not "1m" in imageFile:
            crop = 400
            img[imageFile] = img[imageFile][crop:img[imageFile].shape[0], 0:img[imageFile].shape[1]]
            img_noBaseline[imageFile] = img_noBaseline[imageFile][crop:img_noBaseline[imageFile].shape[0], 0:img_noBaseline[imageFile].shape[1]]
            img_blackNwhite[imageFile] = img_blackNwhite[imageFile][crop:img_blackNwhite[imageFile].shape[0], 0:img_blackNwhite[imageFile].shape[1]]
        
        # Temporary - Crop the right and left of the image
        # If it is 24 or 46, crop
        if "24m" in imageFile or "46m" in imageFile:
            print(Fore.BLUE + "Cropping the right and left of the image")
            crop2 = 600
            img[imageFile] = img[imageFile][0:img[imageFile].shape[0], crop2:img[imageFile].shape[1] - (crop2)]
            img_noBaseline[imageFile] = img_noBaseline[imageFile][0:img_noBaseline[imageFile].shape[0], crop2:img_noBaseline[imageFile].shape[1] - (crop2)]
            img_blackNwhite[imageFile] = img_blackNwhite[imageFile][0:img_blackNwhite[imageFile].shape[0], crop2:img_blackNwhite[imageFile].shape[1] - (crop2)]
        
        """ # Temporary - Crop the right of the image
        crop = 1050
        img[imageFile] = img[imageFile][0:img[imageFile].shape[0], 0:img[imageFile].shape[1] - crop]
        img_noBaseline[imageFile] = img_noBaseline[imageFile][0:img_noBaseline[imageFile].shape[0], 0:img_noBaseline[imageFile].shape[1] - crop]
        img_blackNwhite[imageFile] = img_blackNwhite[imageFile][0:img_blackNwhite[imageFile].shape[0], 0:img_blackNwhite[imageFile].shape[1] - crop]
         """
        # Remove the background avgPixelValue from the image
        if GB == "GB_OFF":
            for i in range(img[imageFile].shape[0]):
                for j in range(img[imageFile].shape[1]):
                    if img[imageFile][i, j] <= avgPixelValue:
                        img_noBaseline[imageFile][i, j] = 0
                    else:
                        img_noBaseline[imageFile][i, j] -= avgPixelValue
        elif GB == "GB_ON":
            for i in range(img[imageFile].shape[0]):
                for j in range(img[imageFile].shape[1]):
                    if img[imageFile][i, j] <= avgPixelValueGB:
                        img_noBaseline[imageFile][i, j] = 0
                    else:
                        img_noBaseline[imageFile][i, j] -= avgPixelValueGB
        
        # Find the max pixel coordinates
        maxPixelCoordinates = np.where(img_noBaseline[imageFile] == np.amax(img_noBaseline[imageFile]))
        maxPixelCoordinatesArray[imageFile] = maxPixelCoordinates
        #print(Fore.YELLOW + "Max pixel coordinates: " + str(maxPixelCoordinates))
        # Get the max pixel value
        maxPixelValue = float(np.amax(img_noBaseline[imageFile]))

        # Make the image color
        img[imageFile] = cv2.cvtColor(img[imageFile], cv2.COLOR_GRAY2RGB)
        img_noBaseline[imageFile] = cv2.cvtColor(img_noBaseline[imageFile], cv2.COLOR_GRAY2RGB)
        img_blackNwhite[imageFile] = cv2.cvtColor(img_blackNwhite[imageFile], cv2.COLOR_GRAY2RGB)
        
        # Count all pixels with (maxPixelValue - treshold) within a XX by XX square around the max pixel and paint them red
        count = 0
        potencySingle = 0
        
        # Paint all the pixels that are above the treshold in white that are also within the square size
        for i in range(maxPixelCoordinates[1][0] - int(squareSize/2), maxPixelCoordinates[1][0] + int(squareSize/2)):
            for j in range(maxPixelCoordinates[0][0] - int(squareSize/2), maxPixelCoordinates[0][0] + int(squareSize/2)):
                if img_noBaseline[imageFile][j, i, 0] >= (maxPixelValue - (maxPixelValue*(treshold/100))):
                    img_blackNwhite[imageFile][j, i] = [255, 255, 255]
                    count += 1
                    potencySingle += img_noBaseline[imageFile][j, i, 0]
        
        # Paint all pixels out of the square size black
        for i in range(img_blackNwhite[imageFile].shape[0]):
            for j in range(img_blackNwhite[imageFile].shape[1]):
                if i < maxPixelCoordinates[0][0] - int(squareSize/2) or i > maxPixelCoordinates[0][0] + int(squareSize/2) or j < maxPixelCoordinates[1][0] - int(squareSize/2) or j > maxPixelCoordinates[1][0] + int(squareSize/2):
                    if img_blackNwhite[imageFile][i, j, 0] == 255:
                        count -= 1
                    img_blackNwhite[imageFile][i, j] = [0, 0, 0]

        avgX = []
        avgY = [] 
        avgDistance = []  
                
        # Paint the non-white pixels black
        for i in range(img_blackNwhite[imageFile].shape[0]):
            for j in range(img_blackNwhite[imageFile].shape[1]):
                if img_blackNwhite[imageFile][i, j, 0] != 255:
                    img_blackNwhite[imageFile][i, j] = [0, 0, 0]
        
        """ # With the white pixels, calculate the average pixels coordinates; 
        # Later calculate from the pixel coordinates obtained, the average distance from the center to all the other white pixels
        for i in range(img_blackNwhite[imageFile].shape[0]):
            for j in range(img_blackNwhite[imageFile].shape[1]):
                if img_blackNwhite[imageFile][i, j, 0] == 255:
                    avgX.append(j)
                    avgY.append(i)
                    avgDistance.append(np.sqrt((j - maxPixelCoordinates[1][0])**2 + (i - maxPixelCoordinates[0][0])**2))
        
        # Calculate the average distance from the center to all the other white pixels
        avgDistance = np.average(avgDistance)
        print(Fore.YELLOW + "Average distance from the center to all the other white pixels: " + str(avgDistance))
        
        # Calculate the average pixel coordinates
        avgX = np.average(avgX)
        avgY = np.average(avgY) """
        
        # Normalize the images to the max pixel value
        #img[imageFile] = cv2.normalize(img[imageFile], None, 0, maxPixelValue, cv2.NORM_MINMAX, cv2.CV_16U)
        
        # Draw a circle around the average pixel coordinates
        #cv2.circle(img_blackNwhite[imageFile], (int(avgX), int(avgY)), int(avgDistance*1.5), (100, 100, 100), 1)

        # Draw a rectangle around the max pixel taking into account the square size
        #cv2.rectangle(img_blackNwhite[imageFile], (maxPixelCoordinates[1][0] - int(squareSize/2), maxPixelCoordinates[0][0] - int(squareSize/2)), (maxPixelCoordinates[1][0] + int(squareSize/2), maxPixelCoordinates[0][0] + int(squareSize/2)), (100, 100, 100), 1)
        
        pixels[k] = count
        maxVal[k] = maxPixelValue
        potencyCnt[k] = potencySingle
        
        # Print the result
        print(Fore.YELLOW + "Max pixel value: " + str(maxPixelValue))
        print(Fore.YELLOW + "Number of pixels above the treshold: " + str(pixels[k]))
        print(Fore.YELLOW + "Sum of all pixel values above the threshold: " + str(potencyCnt[k]))
        k += 1
    cnt = 0
    squareSize = squareSizeBackup
    windowWidth = 200
    
    for imageFile in img:
        cv2.namedWindow(str(distances[cnt]) + "m - Pics", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(str(distances[cnt]) + "m - Pics", windowWidth, windowWidth*3+20*2)
        margin = 0
        
        # Show only part of the image (Being the square around the max pixel)
        img[imageFile] = img[imageFile][maxPixelCoordinatesArray[imageFile][0][0] - int((squareSize/2) + margin):maxPixelCoordinatesArray[imageFile][0][0] + int((squareSize/2) + margin), maxPixelCoordinatesArray[imageFile][1][0] - int((squareSize/2) + margin):maxPixelCoordinatesArray[imageFile][1][0] + int((squareSize/2) + margin)]
        img_noBaseline[imageFile] = img_noBaseline[imageFile][maxPixelCoordinatesArray[imageFile][0][0] - int((squareSize/2) + margin):maxPixelCoordinatesArray[imageFile][0][0] + int((squareSize/2) + margin), maxPixelCoordinatesArray[imageFile][1][0] - int((squareSize/2) + margin):maxPixelCoordinatesArray[imageFile][1][0] + int((squareSize/2) + margin)]
        img_blackNwhite[imageFile] = img_blackNwhite[imageFile][maxPixelCoordinatesArray[imageFile][0][0] - int((squareSize/2) + margin):maxPixelCoordinatesArray[imageFile][0][0] + int((squareSize/2) + margin), maxPixelCoordinatesArray[imageFile][1][0] - int((squareSize/2) + margin):maxPixelCoordinatesArray[imageFile][1][0] + int((squareSize/2) + margin)]
        
        # Add an horizontal line, separating the images, before joining them vertically
        horizontalLine = np.zeros((10, img[imageFile].shape[1], 3), dtype=np.uint8)
        horizontalLine.fill(255)
        
        #Copy 16bit img as a backup
        img_16bit[imageFile] = img[imageFile].copy()
        img_16bitBase[imageFile] = img_noBaseline[imageFile].copy()
        img_16bitBW[imageFile] = img_blackNwhite[imageFile].copy()
        joinedVert_16bit[imageFile] = np.concatenate([img_16bit[imageFile], horizontalLine, img_16bitBase[imageFile], horizontalLine, img_16bitBW[imageFile]], axis=0)
        
        # Convert the images to 8bit
        img[imageFile] = cv2.normalize(img[imageFile], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img_noBaseline[imageFile] = cv2.normalize(img_noBaseline[imageFile], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img_blackNwhite[imageFile] = cv2.normalize(img_blackNwhite[imageFile], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)        
        
        # Join the images vertically
        joinedVert[imageFile] = np.concatenate([img[imageFile], horizontalLine, img_noBaseline[imageFile], horizontalLine, img_blackNwhite[imageFile]], axis=0)
        #joinedVert[imageFile] = np.concatenate((img[imageFile], img_noBaseline[imageFile], img_blackNwhite[imageFile] ), axis=0)
        
        joinedVert[imageFile] = cv2.applyColorMap(joinedVert[imageFile], cv2.COLORMAP_VIRIDIS)
        img_blackNwhite[imageFile] = cv2.applyColorMap(img_blackNwhite[imageFile], cv2.COLORMAP_VIRIDIS)
        img_noBaseline[imageFile] = cv2.applyColorMap(img_noBaseline[imageFile], cv2.COLORMAP_VIRIDIS)
        
        cv2.imshow(str(distances[cnt]) + "m - Pics", joinedVert[imageFile]) 
        
        cv2.setMouseCallback(str(distances[cnt]) + "m - Pics", mouse_callback, joinedVert_16bit[imageFile])
        cnt += 1
        
    cv2.waitKey(0)
    
    
    # Plot the results (Pixels Vs Distance)
    """ plt.plot(distances, pixels)
    plt.plot(distances, maxVal)
    plt.xlabel("Distance (m)")
    plt.ylabel("Number of pixels")
    plt.title("%s; %s; %s; %s" % (POTENCY, EXPOSURE, GB, FOCUS))
    plt.show() """
    
    print(Fore.GREEN + "Done!")
    
    
    # Create a dictionary with the results according to which potencyValues are being used
    # and then save them to a json file
    
    result[POTENCY][EXPOSURE][GB][FOCUS] = pixels.tolist()
    resultMax[POTENCY][EXPOSURE][GB][FOCUS] = maxVal.tolist()
    resultsPotency[POTENCY][EXPOSURE][GB][FOCUS] = potencyCnt.tolist()
    #print(result)
    
    # Save the pics
    for imageFile in img:
        
        # Res up the image
        img_blackNwhite[imageFile] = cv2.resize(img_blackNwhite[imageFile], (img_blackNwhite[imageFile].shape[1]*2, img_blackNwhite[imageFile].shape[0]*2), interpolation=cv2.INTER_NEAREST)
        #cv2.imwrite(subFolderPath[0] + "Results/" + imageFile, img_blackNwhite[imageFile])
        
        #img_noBaseline[imageFile] = cv2.resize(img_noBaseline[imageFile], (img_noBaseline[imageFile].shape[1]*2, img_noBaseline[imageFile].shape[0]*2), interpolation=cv2.INTER_NEAREST)
        #cv2.imwrite(subFolderPath[0] + "Results/" + imageFile + "_noBaseline.png", img_noBaseline[imageFile])
    
    
    if os.path.exists("results.json"):
        
        with open("results.json", "r") as outfile:
            # If the file is empty, dont load any data
            if os.stat("results.json").st_size == 0:
                print("File is empty")
            else:
                data = json.load(outfile)
    else:
        with open("results.json", "w") as outfile:
            json.dump(data, outfile)
    
    if os.path.exists("resultsMax.json"):
        with open("resultsMax.json", "r") as outfile:
            # If the file is empty, dont load any data
            if os.stat("resultsMax.json").st_size == 0:
                print("File is empty")
            else:
                dataMax = json.load(outfile)
    else:
        with open("resultsMax.json", "w") as outfile:
            json.dump(dataMax, outfile)
            
    if os.path.exists("resultsPotency.json"):
        with open("resultsPotency.json", "r") as outfile:
            # If the file is empty, dont load any data
            if os.stat("resultsPotency.json").st_size == 0:
                print("File is empty")
            else:
                dataPotency = json.load(outfile)
    else:
        with open("resultsPotency.json", "w") as outfile:
            json.dump(dataPotency, outfile)
    
    data[POTENCY][EXPOSURE][GB][FOCUS] = result[POTENCY][EXPOSURE][GB][FOCUS]
    dataMax[POTENCY][EXPOSURE][GB][FOCUS] = resultMax[POTENCY][EXPOSURE][GB][FOCUS]
    dataPotency[POTENCY][EXPOSURE][GB][FOCUS] = resultsPotency[POTENCY][EXPOSURE][GB][FOCUS]
    
    with open("results.json", "w+") as outfile:
        #Write the variable back to the file
        #print(data)
        outfile.seek(0)
        json.dump(data, outfile)
    
    with open("resultsMax.json", "w+") as outfile:
        #Write the variable back to the file
        #print(data)
        outfile.seek(0)
        json.dump(dataMax, outfile)
    
    with open("resultsPotency.json", "w+") as outfile:
        #Write the variable back to the file
        #print(data)
        outfile.seek(0)
        json.dump(dataPotency, outfile)
        return

        
    

if __name__ == '__main__':
    main()