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

  def run(self):
      """
      Runs the Streamlit web application.
      """
      st.set_page_config(layout="wide", page_title="Driver Behavior Scoring System")
      st.title("🎯 Driver Behavior Scoring System")

      # Display warning if face detection is not enabled
      if not self.face_detection_enabled:
          st.error("🚨 Face and Eye Detection is disabled. MediaPipe FaceMesh failed to initialize. "
                   "This is often due to an incompatible `mediapipe` or `opencv-python` installation. "
                   "Please try running `pip install --upgrade mediapipe opencv-python` in your terminal.")

      # Initialize models on app startup
      self._initialize_models()

      st.sidebar.header("Controls")
      uploaded_file = st.sidebar.file_uploader("Upload Telemetry Data (CSV)", type=["csv"])
      if uploaded_file:
          self.handle_upload(uploaded_file)

      st.sidebar.markdown("---")
      st.sidebar.header("Real-time Monitoring")
      
      # Only show camera controls if face detection is enabled
      if self.face_detection_enabled:
          start_camera = st.sidebar.button("Start Camera")
          stop_camera = st.sidebar.button("Stop Camera")
      else:
          st.sidebar.info("Camera controls are disabled because face detection failed to initialize.")
          start_camera = False # Disable button logic
          stop_camera = False # Disable button logic


      # Main content area
      col1, col2 = st.columns([2, 1])

      with col1:
          st.subheader("Live Driver Monitoring")
          video_placeholder = st.empty()
          drowsiness_alert_placeholder = st.empty()
          
          # Display current metrics below the video
          st.markdown("---")
          st.subheader("Current Metrics")
          metric_col1, metric_col2, metric_col3 = st.columns(3)
          current_score_placeholder = metric_col1.empty()
          current_risk_placeholder = metric_col2.empty()
          current_ear_placeholder = metric_col3.empty()
          current_blink_placeholder = metric_col3.empty() # Reuse for blink count

      with col2:
          st.subheader("Dashboards")
          st.markdown("---")
          st.subheader("Historical Trends")
          ear_chart_placeholder = st.empty()
          blink_chart_placeholder = st.empty()
          score_chart_placeholder = st.empty()
          risk_chart_placeholder = st.empty()

      st.markdown("---")
      st.subheader("Telemetry Data Analysis")
      if not self.telemetry_data.empty:
          st.write("Uploaded Telemetry Data Overview:")
          st.dataframe(self.telemetry_data.describe())
          
          # Plotting some features from uploaded data
          if 'speed_avg' in self.telemetry_data.columns:
              self.visualizer.plot_feature_distribution(self.telemetry_data, 'speed_avg', "Speed Average Distribution")
          if 'braking_intensity' in self.telemetry_data.columns:
              self.visualizer.plot_feature_distribution(self.telemetry_data, 'braking_intensity', "Braking Intensity Distribution")
          
          # Show clustering results if data is available
          if 'speed_avg' in self.telemetry_data.columns and 'braking_intensity' in self.telemetry_data.columns:
              try:
                  # Ensure features for clustering are available and scaled
                  clustering_features = self.telemetry_data[['speed_avg', 'braking_intensity', 'jerk', 'ear', 'blink_rate']].copy()
                  # Fill any potential NaNs that might arise from feature engineering on small datasets
                  clustering_features = clustering_features.fillna(clustering_features.mean())
                  
                  # Use the scaler from the driver_model
                  # Check if scaler is fitted, if not, fit it with dummy data
                  if not hasattr(self.driver_model.scaler, 'scale_'):
                      st.warning("Scaler not fitted. Fitting with dummy data for clustering plot.")
                      dummy_data_for_scaler = pd.DataFrame({
                          'speed_avg': np.random.rand(10) * 100,
                          'braking_intensity': np.random.rand(10) * 10,
                          'jerk': np.random.randn(10) * 2,
                          'ear': np.random.rand(10) * 0.4 + 0.1,
                          'blink_rate': np.random.randint(0, 10, 10)
                      })
                      self.driver_model.scaler.fit(dummy_data_for_scaler)

                  X_scaled_for_clustering = self.driver_model.scaler.transform(clustering_features)
                  
                  cluster_labels = self.driver_model.clustering_model.predict(X_scaled_for_clustering)
                  self.visualizer.plot_clustering_results(clustering_features, cluster_labels)
              except Exception as e:
                  st.warning(f"Could not plot clustering results: {e}. Ensure data has enough features.")
      else:
          st.info("Upload telemetry data to see detailed analysis and clustering results.")


      if start_camera and self.face_detection_enabled: # Only proceed if camera is started AND face detection is enabled
          self.cap = cv2.VideoCapture(0) # 0 for default webcam
          if not self.cap.isOpened():
              st.error("Error: Could not open webcam. Please ensure it's connected and not in use.")
              self.cap = None
          else:
              st.session_state.run_camera = True
              st.sidebar.success("Camera started!")
      elif start_camera and not self.face_detection_enabled:
          st.error("Cannot start camera: Face and Eye Detection is not initialized.")


      if stop_camera:
          st.session_state.run_camera = False
          if self.cap:
              self.cap.release()
          st.sidebar.warning("Camera stopped.")

      # Loop for real-time processing
      if 'run_camera' not in st.session_state:
          st.session_state.run_camera = False

      if st.session_state.run_camera and self.cap and self.face_detection_enabled: # Add check for face_detection_enabled
          while self.cap.isOpened() and st.session_state.run_camera:
              ret, frame = self.cap.read()
              if not ret:
                  st.error("Failed to grab frame from camera.")
                  st.session_state.run_camera = False
                  break

              processed_frame, current_score, current_risk, drowsy_alert, blink_count = self.process_frame(frame)

              # Update UI elements
              self.visualizer.display_video(processed_frame, video_placeholder)
              current_score_placeholder.metric(label="Driver Safety Score", value=f"{current_score:.1f}/100")
              
              risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
              color_map = {0: "green", 1: "orange", 2: "red"}
              current_risk_placeholder.markdown(f"**Risk Level:** <span style='color:{color_map.get(current_risk, 'black')}; font-weight:bold;'>{risk_map.get(current_risk, 'Unknown')}</span>", unsafe_allow_html=True)
              
              current_ear_placeholder.metric(label="Eye Aspect Ratio (EAR)", value=f"{self.ear_history[-1]:.2f}" if self.ear_history else "N/A")
              current_blink_placeholder.metric(label="Blink Count", value=f"{self.blink_rate_history[-1]}" if self.blink_rate_history else "N/A")

              if drowsy_alert:
                  drowsiness_alert_placeholder.error("🚨 Drowsiness Detected! Take a break.")
              else:
                  drowsiness_alert_placeholder.empty() # Clear alert if not drowsy

              # Update historical charts
              history_df = pd.DataFrame({
                  'EAR': self.ear_history,
                  'Blink Rate': self.blink_rate_history,
                  'Safety Score': self.score_history,
                  'Risk Level': self.risk_history
              })
              history_df['Time'] = range(len(history_df)) # Simple time index

              if not history_df.empty:
                  self.visualizer.plot_time_series(history_df, 'EAR', "EAR Over Time")
                  self.visualizer.plot_time_series(history_df, 'Blink Rate', "Blink Rate Over Time")
                  self.visualizer.plot_time_series(history_df, 'Safety Score', "Safety Score Over Time")
                  self.visualizer.plot_time_series(history_df, 'Risk Level', "Risk Level Over Time")

              time.sleep(0.05) # Control frame rate

          if self.cap:
              self.cap.release()
          st.sidebar.warning("Camera stream ended.")
      elif st.session_state.run_camera and not self.face_detection_enabled:
          st.warning("Camera stream cannot start because Face and Eye Detection is not initialized.")

if __name__ == "__main__":
  app = DriverBehaviorApp()
  app.run()
