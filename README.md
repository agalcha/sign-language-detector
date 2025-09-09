# sign-language-detector

A customizable real-time sign language detection tool built with Python, OpenCV, and MediaPipe.  
Train your own signs (any gestures you like) and watch the model recognize them in real-time.  
Perfect for experimenting with CV, ML, and interactive AI applications.  

# **🌟 Features**  
📸 Live Data Collection – Capture your own hand gestures directly from your webcam.  
🧠 Custom Training – Train the model on any set of signs you define.  
⚡ Real-Time Prediction – Detect gestures on the fly with confidence scores.  
🎛️ Interactive UI – Control data collection, training, and detection through Colab buttons.  
🛠️ Lightweight ML – Uses a simple but effective KNN classifier under the hood.  

# **🖥️ Demo (in Colab)**  
Clone or open this notebook in Google Colab.  
Run the setup cells (imports + environment).  
**Use the buttons in the Control Panel:**  
✍️ Start Data Collection → Capture your gestures.  
🧑‍🏫 Train Model → Build a classifier for your signs.  
🔍 Start Detection → Predict signs in real-time.  

# **⚙️ Tech Stack**  
Python 3 🐍  
OpenCV 🎥  
MediaPipe 🖐️  
scikit-learn 🤖  
Google Colab ☁️  

# **🚀 How It Works**  
Hand Tracking → MediaPipe extracts 21 key hand landmarks.  
Feature Engineering → Normalized (x, y) coordinates become 42-dim vectors.  
Training → A KNN classifier learns to map gestures → labels.  
Prediction → Capture a gesture → classify → display result with confidence %.  

# **📂 Project Structure**  
├── asl_data/          # Collected CSV samples for each sign  
├── sign_language.py   # Core detection + training logic  
└── README.md          # You're here!   

# **💡 Ideas to Extend**
Add CNNs or RNNs for higher accuracy.  
Support multi-hand or multi-sign detection.  
Build a web app interface (e.g., Streamlit/Flask).  
Export the model for mobile apps.  

# **🪄 Example Output**  
👉 Prediction: I_Love_You (100%)  
✨ Have fun teaching your computer new signs!  
