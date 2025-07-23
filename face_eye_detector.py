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
  
  def calculate_EAR(self, landmarks, eye_indices):
    if not landmarks or len(landmarks) < max(eye_indices) + 1:
        return 0.0
    try:
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]
        A = self._euclidean_distance(p2, p6)
        B = self._euclidean_distance(p3, p5)
        C = self._euclidean_distance(p1, p4)
        ear = (A + B) / (2.0 * C)
        return ear
    except IndexError:
        return 0.0

  def detect_landmarks(self, frame):
    if not self.initialized:
        return frame, None
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = self.face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style())
    return image, results

  def detect_drowsiness(self, ear):
    if ear < self.EAR_THRESHOLD:
        self.COUNTER += 1
        if self.COUNTER >= self.CONSECUTIVE_FRAMES:
            self.DROWSY_ALERT = True
    else:
        self.COUNTER = 0
        self.DROWSY_ALERT = False
    return self.DROWSY_ALERT

  def detect_blink(self, ear):
    blink_detected = False
    if self.LAST_EAR > self.EAR_THRESHOLD and ear < self.EAR_THRESHOLD:
        pass
    elif self.LAST_EAR < self.EAR_THRESHOLD and ear > self.EAR_THRESHOLD:
        self.BLINK_COUNTER += 1
        blink_detected = True
    self.LAST_EAR = ear
    return blink_detected

  def process_frame(self, frame):
    if not self.initialized:
        return frame, 0.0, False, 0
    processed_frame, results = self.detect_landmarks(frame)
    ear = 0.0
    drowsy_alert = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = self.calculate_EAR(face_landmarks.landmark, self.LEFT_EYE_EAR_INDICES)
            right_ear = self.calculate_EAR(face_landmarks.landmark, self.RIGHT_EYE_EAR_INDICES)
            ear = (left_ear + right_ear) / 2.0
            drowsy_alert = self.detect_drowsiness(ear)
            self.detect_blink(ear)
    return processed_frame, ear, drowsy_alert, self.BLINK_COUNTER
