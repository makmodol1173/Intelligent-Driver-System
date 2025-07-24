# AI-Powered Driver Safety Monitor with OpenCV

## 🎯 Project Goal
This project designs and develops a modular Driver Behavior Scoring System that processes driving telemetry data, performs real-time face and eye detection (for drowsiness/distraction) using OpenCV and MediaPipe, and uses machine learning to score driver safety and risk levels. All results are presented in an interactive Streamlit web application.

## ✨ Key Features
*   **Telemetry Data Processing:** Load, preprocess, and engineer features from driving data (speed, braking, acceleration).
*   **Real-time Face & Eye Detection:** Utilize MediaPipe for live webcam feed to detect facial landmarks, calculate Eye Aspect Ratio (EAR), and identify signs of drowsiness.
*   **Machine Learning Models:** Employ regression, classification, and clustering models (scikit-learn) to predict driver safety scores and classify risk levels.
*   **Interactive Streamlit App:** A user-friendly web interface displaying live video, real-time scores, drowsiness alerts, and historical trend dashboards.
*   **Audio Alerts:** Implements distinct audio alerts. A continuous high-risk alert sounds specifically when drowsiness (prolonged eye closure) is detected.
*   **Object-Oriented Design:** Core functionality implemented using OOP principles for clarity and extensibility.

## 🛠️ Technologies & Libraries
*   Python 3.x
*   Streamlit (interactive web UI)
*   OpenCV (cv2) & MediaPipe (facial landmark detection)
*   scikit-learn, Pandas, NumPy (ML & data handling)
*   Plotly (charts/graphs)

## 🚀 Setup and Installation

Follow these steps to get the project up and running on your local machine:

1.  **Clone the Repository:**
    \`\`\`bash
    git clone <repository_url>
    cd <project_directory>
    \`\`\`
    (Replace `<repository_url>` and `<project_directory>` with your actual repository URL and the name of the cloned directory.)

2.  **Create a Virtual Environment (Recommended):**
    \`\`\`bash
    python -m venv venv
    \`\`\`

3.  **Activate the Virtual Environment:**
    *   **Windows:**
        \`\`\`bash
        .\venv\Scripts\activate
        \`\`\`
    *   **macOS/Linux:**
        \`\`\`bash
        source venv/bin/activate
        \`\`\`

4.  **Install Dependencies:**
    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`

    **Troubleshooting MediaPipe:**
    If you encounter a `RuntimeError` related to MediaPipe, try reinstalling:
    \`\`\`bash
    pip uninstall mediapipe opencv-python -y
    pip install --upgrade mediapipe opencv-python protobuf
    \`\`\`

5.  **Run the Streamlit Application:**
    \`\`\`bash
    streamlit run app.py
    \`\`\`
    This command will open the application in your default web browser.

## 💡 Usage
*   **Upload Telemetry Data:** Use the sidebar to upload a CSV file containing driving telemetry data (e.g., `data/dataset.csv`).
*   **Start Camera:** Click "Start Camera" in the sidebar to activate your webcam for real-time face and eye detection.
*   **Monitor Dashboards:** Observe live safety scores, risk classifications, drowsiness alerts, and historical trends on the main dashboard.

## 📞 Contact Me
For any questions or feedback, feel free to reach out:
*   **Email:** [makmodol1173@gmail.com](mailto:makmodol1173@gmail.com)
*   **GitHub:** [github.com/makmodol1173](https://github.com/makmodol1173)
