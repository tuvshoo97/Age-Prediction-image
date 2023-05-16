import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from fastai.vision.all import *
import gdown
import os

from twilio.rest import Client

# Check if the Haar Cascade XML file exists, otherwise download it
xml_file_path = "haarcascade_frontalface_default.xml"
if not os.path.isfile(xml_file_path):
    with st.spinner("Downloading Haar Cascade XML file..."):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        gdown.download(url, xml_file_path, quiet=False)

# Load the Haar Cascade Classifier for face detection
try:
    face_cascade = cv2.CascadeClassifier(xml_file_path)
except Exception:
    st.write("Error loading cascade classifiers")
# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = os.environ['TWILIO_ACCOUNT_SID'] = 'ACdabf91e1a09e6a08159e78171d6639c2'
auth_token = os.environ['TWILIO_AUTH_TOKEN'] = '88adea9a008d68df050dfff876ce65e2'

client = Client(account_sid, auth_token)

token = client.tokens.create()

def get_x(row):
    return row['image']

def get_y(row):
        return row['age']

model_path = "export.pkl"
if not os.path.isfile(model_path):
    with st.spinner("Downloading model... this may take a while! \n Don't stop it!"):
        url = 'https://drive.google.com/uc?id=1gJNYV3KB_oeS7scI9lpQIfSuj-Lb9Og0'
        gdown.download(url, model_path, quiet=False)
    learn = load_learner(model_path)
else:
    learn = load_learner(model_path)
    
class AgeDetector(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to grayscale for face detection
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Perform face detection using the Haar Cascade Classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around each detected face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the region of interest (ROI) or the cropped face image
            cropped_face = img[y:y + h, x:x + w]

            # Perform age detection on the cropped face image using your custom age detection algorithm
            age = learn.predict(cropped_face)[0][0]

            # Display the predicted age on the frame
            age_text = "Age: {}".format(round(age, 0))
            cv2.putText(img, age_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            

        return img

def main():
    #st.set_page_config(page_title="Age Detection", layout="wide")

    st.markdown("""# Real-Time Age Prediction with Webcam Access

    Using    this app, you can predict the age of a person in real-time by accessing your webcam. Impress your friends with this cutting-edge deep learning technology!

    This app was created as a project for the Deep Learning course at LETU Mongolia American University. Have fun exploring the world of age detection with live video!""")

    # Configure the Streamlit WebRTC component
    webrtc_ctx = webrtc_streamer(key="example" ,mode=WebRtcMode.SENDRECV,rtc_configuration={"iceServers": token.ice_servers},
                                 video_processor_factory=AgeDetector)

if __name__ == "__main__":
    main()
