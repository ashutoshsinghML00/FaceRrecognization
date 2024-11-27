import os
import cv2
import numpy as np
import tensorflow as tf
from kivy.app import App
from layers import L1_dist
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture

# Build app and layout 
class CamApp(App):

    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Un-initiated", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model(r'C:\Users\srashti\OneDrive\Desktop\My_Face_Project\My_Face_Project\app\siamese_network.h5', custom_objects={'L1_dist':L1_dist})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0) # 0.03030
        
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring() # filiping the frame and convert to string
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 105x105px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 105x105x3
        img = tf.image.resize(img, (105,105))
        # Scale image to be between 0 and 1 
        img = img / 255.0
        
        # Return image
        return img

    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.5 # before prediction, for positives
        verification_threshold = 0.5 # proposition of pre need to be positive.

        # Capture input image from our webcam
        SAVE_PATH = os.path.join(r'app\application_data\input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # for image in (os.listdir(r"app\application_data\verification_images")):
        #     input_img = self.preprocess(r'app\application_data\input_image\input_image.jpg')
        #     location = r'app\application_data\verification_images'+ str(image)
        #     validation_img = self.preprocess(location)
            


        # Build results array
        results = []
        for image in os.listdir(os.path.join(r'app\application_data\verification_images')):
            input_img = self.preprocess(os.path.join(r'app\application_data\input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join(r'app\application_data\verification_images', image))


            
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join(r'app\application_data\verification_images'))) 
        verified = verification > verification_threshold

        # Set verification text 
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        # Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        
        return results, verified


if __name__ == '__main__':
    CamApp().run()
