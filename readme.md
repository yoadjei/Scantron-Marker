# KNUST Scantron Marker

## Overview
This Python application fulfills the requirements for the CE 257 Computing Project to develop an application for marking Scantron/Scannable sheets. The program processes images of scannable sheets, identifies marked answers, compares them against an answer key and generates performance reports.

## Features
- Processes images of KNUST scannable sheets
- Detects marked bubbles using computer vision techniques
- Maps detected marks to questions and answer options
- Compares student answers to a predefined answer key
- Calculates scores and provides detailed grading
- Generates a summary report of student performance by question

## Requirements
- Python 3.6+
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pandas


## Installation
```bash
pip install opencv-python numpy matplotlib pandas
```


## Usage
1. Place the script in the same directory as your scannable sheet image(s)
2. Run the script:
   ```
   python scantron_marker.py
   ```
3. The script will process each sheet and generate visualizations and reports

## How It Works
1. The program loads and preprocesses the scantron sheet image
2. It applies thresholding to identify dark markings on the sheet
3. Contours are detected and filtered to find bubble shapes
4. Marked bubbles are identified based on pixel intensity
5. The program maps these marked bubbles to questions and answer options
6. Student answers are compared to the answer key for grading
7. Results are visualized and a summary report is generated

## Output
- Visualized graded sheets (JPG images)
- Question performance summary (PDF chart)
- Detailed performance data (CSV file)
