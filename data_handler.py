import pandas as pd
import numpy as np

class DataHandler:
  """
  Handles loading, cleaning, and preprocessing of driving telemetry data,
  and performs feature engineering.
  """
  def __init__(self):
      self.data = None

  def load_data(self, file_path):
      """
      Loads driving telemetry data from a CSV file.
      Args:
          file_path (str): Path to the CSV file.
      Returns:
          pd.DataFrame: Loaded DataFrame.
      """
      try:
          self.data = pd.read_csv(file_path)
          print(f"Data loaded successfully from {file_path}")
          return self.data
      except Exception as e:
          print(f"Error loading data: {e}")
          return None

  def preprocess(self, df):
      """
      Performs basic preprocessing on the DataFrame.
      - Fills missing numerical values with the mean.
      - Ensures 'speed', 'braking', 'acceleration' columns exist (dummy if not).
      Args:
          df (pd.DataFrame): Input DataFrame.
      Returns:
          pd.DataFrame: Preprocessed DataFrame.
      """
      if df is None:
          return None

      # For demonstration, ensure basic columns exist or create dummy ones
      if 'speed' not in df.columns:
          df['speed'] = np.random.rand(len(df)) * 120 # km/h
      if 'braking' not in df.columns:
          df['braking'] = np.random.rand(len(df)) * 10 # 0-10 scale
      if 'acceleration' not in df.columns:
          df['acceleration'] = np.random.rand(len(df)) * 5 # m/s^2

      # Fill missing numerical values with mean
      for col in df.select_dtypes(include=np.number).columns:
          if df[col].isnull().any():
              df[col] = df[col].fillna(df[col].mean())
      
      print("Data preprocessing complete.")
      return df

  def feature_engineering(self, df):
      """
      Engineers new features from the raw telemetry data.
      Args:
          df (pd.DataFrame): Preprocessed DataFrame.
      Returns:
          pd.DataFrame: DataFrame with engineered features.
      """
      if df is None:
          return None

      # Example feature engineering
      df['speed_avg'] = df['speed'].rolling(window=5, min_periods=1).mean()
      df['braking_intensity'] = df['braking'].apply(lambda x: 1 if x > 5 else 0) # Binary: hard braking
      df['jerk'] = df['acceleration'].diff().fillna(0) # Change in acceleration

      # Create a dummy 'driving_style' for clustering if not present
      if 'driving_style' not in df.columns:
          df['driving_style'] = np.random.choice(['aggressive', 'normal', 'calm'], size=len(df))

      print("Feature engineering complete.")
      return df
