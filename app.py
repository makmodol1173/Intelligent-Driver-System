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

  def _initialize_models(self):
      """Initializes or loads ML models."""
      if not self.driver_model.load_models():
          st.info("Training models with dummy data. This may take a moment...")
          # Create dummy data for initial model training
          dummy_data = pd.DataFrame({
              'speed': np.random.rand(100) * 120,
              'braking': np.random.rand(100) * 10,
              'acceleration': np.random.rand(100) * 5,
              'ear': np.random.rand(100) * 0.4 + 0.1, # Simulate EAR values
              'blink_rate': np.random.randint(0, 10, 100), # Simulate blink rates
              'risk_level': np.random.randint(0, 3, 100) # 0: Low, 1: Medium, 2: High
          })
          dummy_data = self.data_handler.preprocess(dummy_data)
          dummy_data = self.data_handler.feature_engineering(dummy_data)
          
          # Ensure 'ear' and 'blink_rate' are present in the engineered features for training
          # The feature_engineering method doesn't add these, so add them here if not present
          if 'ear' not in dummy_data.columns:
              dummy_data['ear'] = np.random.rand(len(dummy_data)) * 0.4 + 0.1
          if 'blink_rate' not in dummy_data.columns:
              dummy_data['blink_rate'] = np.random.randint(0, 10, len(dummy_data))

          self.driver_model.train_models(dummy_data)
          # self.driver_model.save_models() # Uncomment to save models after training

  def handle_upload(self, uploaded_file):
      """
      Handles uploaded telemetry data, preprocesses, and engineers features.
      Args:
          uploaded_file (UploadedFile): Streamlit uploaded file object.
      """
      if uploaded_file is not None:
          self.telemetry_data = self.data_handler.load_data(uploaded_file)
          if self.telemetry_data is not None:
              self.telemetry_data = self.data_handler.preprocess(self.telemetry_data)
              self.telemetry_data = self.data_handler.feature_engineering(self.telemetry_data)
              st.success("Telemetry data processed successfully!")
              st.dataframe(self.telemetry_data.head())
          else:
              st.error("Failed to process uploaded file.")

  def process_frame(self, frame):
      """
      Processes a single video frame for face/eye detection,
      combines with dummy telemetry features, and predicts driver state.
      Args:
          frame (np.array): Current video frame.
      Returns:
          tuple: (processed_frame, current_score, current_risk, drowsy_alert, blink_count)
      """
      processed_frame, ear, drowsy_alert, blink_count = self.face_eye_detector.process_frame(frame)

      # Simulate real-time telemetry features (replace with actual sensor data if available)
      # For simplicity, we'll use random values for speed, braking, acceleration
      # and combine them with the detected EAR and blink rate.
      current_telemetry_features = {
          'speed_avg': np.random.rand() * 80 + 20, # 20-100 km/h
          'braking_intensity': np.random.rand() * 8, # 0-8
          'jerk': np.random.randn() * 2, # -2 to 2
          'ear': ear, # Use detected EAR
          'blink_rate': blink_count # Use detected blink count
      }
      
      # Convert to DataFrame for consistent input to DriverModel
      current_features_df = pd.DataFrame([current_telemetry_features])

      # Predict score and risk
      current_score = self.driver_model.predict_score(current_features_df)
      current_risk = self.driver_model.classify_risk(current_features_df)

      # Update history for plotting
      self.ear_history.append(ear)
      self.blink_rate_history.append(blink_count)
      self.score_history.append(current_score)
      self.risk_history.append(current_risk)

      # Keep history to a reasonable length
      max_history_len = 100
      self.ear_history = self.ear_history[-max_history_len:]
      self.blink_rate_history = self.blink_rate_history[-max_history_len:]
      self.score_history = self.score_history[-max_history_len:]
      self.risk_history = self.risk_history[-max_history_len:]

      return processed_frame, current_score, current_risk, drowsy_alert, blink_count
if __name__ == "__main__":
  app = DriverBehaviorApp()
  app.run()
