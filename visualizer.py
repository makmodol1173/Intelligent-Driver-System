import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np

class Visualizer:
  """
  Creates and updates Streamlit UI components and visualizations.
  """
  def __init__(self):
      pass

  def display_video(self, frame, placeholder):
      """
      Displays a video frame in Streamlit.
      Args:
          frame (np.array): The video frame to display.
          placeholder (st.empty): Streamlit placeholder to update the image.
      """
      if frame is not None:
          placeholder.image(frame, channels="BGR", use_column_width=True)

  def plot_time_series(self, data, value_col, title="Time Series Plot"):
      """
      Plots a time series of a given value.
      Args:
          data (pd.DataFrame): DataFrame containing the time series data.
          value_col (str): Name of the column to plot.
          title (str): Title of the plot.
      """
      if data is None or data.empty or value_col not in data.columns:
          st.warning(f"No data or '{value_col}' column found for time series plot.")
          return

      # Ensure 'timestamp' or an index is available for plotting
      if 'Time' not in data.columns: # Use 'Time' column from app.py history_df
          data['Time'] = pd.to_datetime(data.index) # Fallback to index if 'Time' not present

      fig = px.line(data, x='Time', y=value_col, title=title)
      st.plotly_chart(fig, use_container_width=True)

  def show_driver_score(self, score):
      """
      Displays the driver safety score using a Streamlit metric.
      Args:
          score (float): The driver safety score.
      """
      st.metric(label="Driver Safety Score", value=f"{score:.1f}/100")

  def show_risk_classification(self, risk_level):
      """
      Displays the driver risk classification.
      Args:
          risk_level (int): The predicted risk level (0: Low, 1: Medium, 2: High).
      """
      risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
      color_map = {0: "green", 1: "orange", 2: "red"}
      
      st.markdown(f"**Risk Level:** <span style='color:{color_map.get(risk_level, 'black')}; font-weight:bold;'>{risk_map.get(risk_level, 'Unknown')}</span>", unsafe_allow_html=True)

  def show_alerts(self, alert_msg, alert_type="warning"):
      """
      Displays alerts in Streamlit.
      Args:
          alert_msg (str): The message to display.
          alert_type (str): Type of alert ('info', 'warning', 'error', 'success').
      """
      if alert_type == "info":
          st.info(alert_msg)
      elif alert_type == "warning":
          st.warning(alert_msg)
      elif alert_type == "error":
          st.error(alert_msg)
      elif alert_type == "success":
          st.success(alert_msg)
      else:
          st.write(alert_msg)
  def plot_clustering_results(self, data, cluster_labels):
      if data is None or data.empty or cluster_labels is None or len(cluster_labels) == 0:
          st.warning("No data or cluster labels for plotting clustering results.")
          return

      data_with_clusters = data.copy()
      data_with_clusters['Cluster'] = cluster_labels.astype(str)

      features_to_plot = [col for col in ['speed_avg', 'braking_intensity', 'jerk', 'ear', 'blink_rate']
                          if col in data_with_clusters.columns]

      if len(features_to_plot) >= 2:
          fig = px.scatter(data_with_clusters, x=features_to_plot[0], y=features_to_plot[1], color='Cluster',
                           title='Driver Clusters (Sample Features)', hover_data=data_with_clusters.columns)
          st.plotly_chart(fig, use_container_width=True)
      elif len(features_to_plot) == 1:
          st.write(f"Cannot plot 2D scatter for clustering with only one feature: {features_to_plot[0]}")
          st.dataframe(data_with_clusters[['Cluster', features_to_plot[0]]].head())
      else:
          st.warning("Not enough features to plot clustering results.")
          st.dataframe(data_with_clusters[['Cluster']].head())

  def plot_feature_distribution(self, data, feature_col, title="Feature Distribution"):
      if data is None or data.empty or feature_col not in data.columns:
          st.warning(f"No data or '{feature_col}' column found for distribution plot.")
          return
      
      fig = px.histogram(data, x=feature_col, title=title)
      st.plotly_chart(fig, use_container_width=True)
