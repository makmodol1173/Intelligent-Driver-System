import streamlit as st
import cv2
import pandas as pd
import numpy as np
import time

# Import the custom classes
from data_handler import DataHandler
from face_eye_detector import FaceEyeDetector
from driver_model import DriverModel
from visualizer import Visualizer

class DriverBehaviorApp:
  """
  Main controller class for the Driver Behavior Scoring System Streamlit app.
  Coordinates data flow, manages app lifecycle, and user interactions.
  """
  def __init__(self):
      self.data_handler = DataHandler()
      self.face_eye_detector = FaceEyeDetector()
      self.driver_model = DriverModel()
      self.visualizer = Visualizer()
      self.cap = None # OpenCV VideoCapture object
      self.telemetry_data = pd.DataFrame()
      self.realtime_features = pd.DataFrame(columns=['speed_avg', 'braking_intensity', 'jerk', 'ear', 'blink_rate'])
      self.blink_rate_history = []
      self.ear_history = []
      self.score_history = []
      self.risk_history = []
      self.face_detection_enabled = self.face_eye_detector.initialized # Check if FaceEyeDetector initialized successfully
