import json
import pprint as pp

import cv2
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style, init

# Read Json file and format it to pprint
""" with open('results.json',"r") as f:
    data = json.load(f)
    
with open('resultsPotency.json',"r") as f:
    dataP = json.load(f)
    
with open('resultsMax.json',"r") as f:
    dataM = json.load(f)

with open('results_pprint.json',"w+") as f:
    pp.pprint(data,f)
    
with open('resultsPotency_pprint.json',"w+") as f:
    pp.pprint(dataP,f)

with open('resultsMax_pprint.json',"w+") as f:
    pp.pprint(dataM,f) """
    
    
# Read Json file and format it to make it easier to copy and paste to Excel
def format_output(data):
    output = ""
    for mW, mW_data in data.items():
        for us, us_data in mW_data.items():
            for gb, gb_data in us_data.items():
                for f, f_data in gb_data.items():
                    output += f"{mW}_{us} {gb} {f}\n"
                    for i, value in enumerate(f_data, start=1):
                        output += f"{value}\n"
                        if i % 4 == 0:
                            output += "\n"
    return output

with open('results.json',"r") as f:
    data = json.load(f)

with open('resultsMax.json',"r") as f:
    dataM = json.load(f)

with open('resultsPotency.json',"r") as f:
    dataP = json.load(f)
    
data_output = format_output(data)
data_outputM = format_output(dataM)
data_outputP = format_output(dataP)

with open('results_pprint.txt',"w+") as f:
    f.write(data_output)
    
with open('resultsMax_pprint.txt',"w+") as f:
    f.write(data_outputM)

with open('resultsPotency_pprint.txt',"w+") as f:
    f.write(data_outputP)
    
