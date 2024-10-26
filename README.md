
<br />

<h3 align="center">LiDART Results Analyzer</h3>

  <p align="center">
    A list of scripts that give the user the necessary means to analyze the pictures taken by LiDART Board Control script.
    
  </p>
</div>


<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>

## Getting Started

All scripts included in this repository were used with Python 3.9.13 in a virtual environment managed by Python's own 'venv'. The following instructions will cover all dependencies along with some extra configurations needed to make things work.

Instructions will be for Windows 10 or 11, replication for Ubuntu or any other Linux distribution shall be adjusted accordingly.

### Prerequisites

To start, make sure Python 3.9.13 is installed (Usage of other versions might have unintended side effects, but possibly shouldn't interfere much). For this, you can go to [Python's downloads page](https://www.python.org/downloads/) and download the correct version. Also check [Python's documentation](https://wiki.python.org/moin/BeginnersGuide/Download) if unsure how to install.

This repository will not cover how to setup a virtual environment (venv), but the user is recommended to create one and use it for this project.

For this scripts there's no need to have the camera's drivers installed.

### Installation

Once Python is installed, there's a few libraries that need to be installed. For that we can use the ``requirements.txt`` provided. 
Open a terminal on the projects folder and use the following command:

```bash
pip install -r requirements.txt
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

We can break down this repository into 3 parts:

- Image Reading / Modifying:
	- ``imageAnalyzer.py``: This script is responsible for analyzing the resulting pictures of a script called ``boardControl.py`` present in the ``LiDART Board Control`` repo.
	- ``imageMerger.py``: When using various pictures with the same attributes for a more 'accurate' value, this script can be used to merge all the corresponding pictures into a single one.
	- ``maxImageBit.py``: Check where the max pixel value is located and present an histogram of all pixel values.
	- ``merger.py``: It was used to merge all baselines pictures into one. Deprecated method and not factually right for data analysis.
- Result formatter:
	- ``reformatJsons.py``: After using the ``imageAnalyzer.py`` script, some ``.json`` files will be created. This script formats them for an easier review of all the values to further enhance transferring values into Excel for example.
- Results:
	- ``results.json``: List of Nº of pixels that fall into a certain threshold value according to the max pixel value;
	- ``resultsMax.json``: List of maximum pixel value;
	- ``resultsPotency.json``: The sum of all pixels that fall into the threshold, which is an indirect way to see the laser's power.

<p align="right">(<a href="#readme-top">back to top</a>)</p>