import cv2
import mediapipe as mp
import numpy as np
import time

class FaceEyeDetector:
  """
  Performs real-time face and eye detection using MediaPipe,
  calculates Eye Aspect Ratio (EAR), and detects drowsiness.
  """
  def __init__(self):
      self.mp_face_mesh = mp.solutions.face_mesh
      self.face_mesh = None # Initialize to None
      self.initialized = False # Flag to indicate successful initialization

      try:
          self.face_mesh = self.mp_face_mesh.FaceMesh(
              max_num_faces=1,
              refine_landmarks=True,
              min_detection_confidence=0.5,
              min_tracking_confidence=0.5
          )
          self.initialized = True
          print("MediaPipe FaceMesh initialized successfully.")
      except Exception as e:
          print(f"Error initializing MediaPipe FaceMesh: {e}")
          print("Face and eye detection will be disabled.")
          self.initialized = False

      if self.initialized:
          self.mp_drawing = mp.solutions.drawing_utils
          self.mp_drawing_styles = mp.solutions.drawing_styles

          # MediaPipe indices for a 6-point EAR model (approximate, based on common usage):
          # These indices are for the 468-point FaceMesh model.
          # Left eye: P1, P2, P3, P4, P5, P6 (horizontal, vertical)
          self.LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
          # Right eye: P1, P2, P3, P4, P5, P6 (horizontal, vertical)
          self.RIGHT_EYE_EAR_INDICES = [33, 160, 158, 133, 145, 153]

          # Drowsiness detection thresholds
          self.EAR_THRESHOLD = 0.25
          self.CONSECUTIVE_FRAMES = 20 # Number of consecutive frames below EAR_THRESHOLD
          self.COUNTER = 0
          self.DROWSY_ALERT = False

          # Blink detection
          self.BLINK_COUNTER = 0
          self.LAST_EAR = 0
          self.BLINK_THRESHOLD = 0.05 # EAR difference to detect a blink (smaller value for more sensitivity)

  def _euclidean_distance(self, p1, p2):
      """Calculates the Euclidean distance between two points."""
      return np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
