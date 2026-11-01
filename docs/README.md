# Intelligent Traffic Management System (YOLOv8)

An AI-based traffic signal controller that detects vehicles in each lane,
measures traffic density, and adjusts green light timing automatically.
An ambulance detection model gives emergency vehicles priority.

## Features
- Real-time vehicle detection and lane-wise counting (YOLOv8 + OpenCV)
- Density-based green light timing (10 s + 2 s per vehicle, max 40 s)
- Ambulance detection with signal override (custom-trained model)
- Flask web dashboard with live video, lane counts and signal state
- User login with hashed passwords and admin/user roles (SQLite)
- Traffic analysis chart and CSV report download

## Tech stack
Python 3.11, YOLOv8 (ultralytics), OpenCV, Flask, SQLite, Chart.js

## How to run
1. Install dependencies:
   pip install ultralytics opencv-python flask
2. Put a traffic video named traffic.mp4 in the project folder.
3. Put your trained ambulance model best.pt in the project folder.
4. Run:
   python app.py
5. Open http://127.0.0.1:5000, register an account (the first account
   becomes admin) and log in.

## Project structure
app.py             Flask app, detection loop, routes
traffic_logic.py   Signal controller
templates/         HTML pages (login, register, dashboard, analysis)

## Status and known issues
Working: vehicle detection, lane counts, signal logic, login, dashboard,
analysis chart, CSV report.
Known issues:
- Uploading a new video does not switch the live feed yet.
- The ambulance override needs more testing with real ambulance videos.

## Future work
Connect to real signal hardware, support multiple junctions, add vehicle
tracking, and measure accuracy on a labelled test set.

## Author
Kesanakurthi Jahnavi, CSE (AI and ML)