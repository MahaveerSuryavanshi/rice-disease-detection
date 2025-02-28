import streamlit as st
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
# model prediction

def prediction(pred_image):
  model = load_model('MobileNet_model.h5')  # Ensure correct loading

  if pred_image is None:
    return None  # Handle case where no image is uploaded

  # Resize the image correctly to (224, 224)
  image = load_img(pred_image, target_size=(224, 224))  # ✅ Fixed to 224x224
  image_array = img_to_array(image)  # Convert to NumPy array
  image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
  # Normalize image (if your model expects it)
  image_array = image_array / 255.0  # ✅ Ensure values are between 0 and 1
  # Model prediction
  prediction = model.predict(image_array)
  result_index = np.argmax(prediction)
  return result_index

st.sidebar.title('Dashboard')
app_page_select = st.sidebar.selectbox("Select Page ",['Home','App','About','Contact']) 
st.sidebar.markdown("""
                    
- Design and Developed By Mahaveer
- Developed using the Deep Learning CNN and Transfer Learning


""")
if(app_page_select =="Home"):
  
  st.header("Rice Disease Detection System")
  image_path ="rice-disease.png"
  st.image(image_path,use_container_width =True)
  st.markdown("""
      ## 🌾 Early Rice Disease Detection Using AI
      ### Protect Your Crops, Maximize Your Yield

      Rice farming is the backbone of global food security, but crop diseases can lead to devastating losses. At Our Platform, we empower farmers with AI-driven technology to detect rice diseases early—helping you take action before it’s too late.


      ### AI-Powered Rice Disease Detection
      Using advanced Deep Learning (CNN - MobileNet), our platform accurately identifies four major rice diseases:
      
      - ✅ Bacterial Blight – Causes wilting and drying of leaves.
      - ✅ Blast – A fungal disease that creates lesions and reduces yield.
      - ✅ Brown Spot – Affects leaves, leading to poor grain quality.
      - ✅ Tungro – A viral disease causing yellowing and stunted growth.

      - ✅ **Instant Disease Detection** – Upload a leaf image and get results in seconds.  
      - ✅ **Cloud-Based AI Model** – Powered by CNN and transfer learning for high accuracy.  
      - ✅ **Comprehensive Disease Coverage** – Detects Bacterial Blight, Blast, Brown Spot, and  Tungro.  
      - ✅ **Accessible Anytime, Anywhere** – Works seamlessly on mobile and desktop.  

      - 🔍 **Start diagnosing now! Click below to use our web app.**  
    ### How It Works
    - 📸 Upload an Image – Take a photo of the affected rice plant.
    - 🤖 AI Analysis – Our deep learning model detects the disease.
    - 📊 Get Instant Results – Receive disease insights and prevention tips.
        
     ***Start Protecting Your Rice Crops Today!
      Take control of your harvest with smart technology. Detect rice diseases early and safeguard your yield.***
  """)

elif(app_page_select=="App"):
  image_path ="rice-disease.png"
  st.image(image_path,use_container_width =True)
  st.header("App - Page")
  test_image =st.file_uploader("Choose and image - up to 200MB")
  if(st.button("Show Image")):
    st.image(test_image,use_container_width=True)
  if(st.button("Predict the image")):

    st.spinner()
    time.sleep(2)
    classes =['Bacterialblight','Blast','Brownspot','Tungro']
    result_index = prediction(test_image)
    st.success("Model it is predicted {}".format(classes[result_index]))


elif(app_page_select=="About"):
  st.header("About - Page")
 
  st.markdown("""
  ----------------------------------
  ### 🌾 Empowering Farmers with Early Rice Disease Detection

    Welcome to out platform, where technology meets agriculture to protect one of the world’s most essential crops—rice. We are dedicated to helping farmers detect and manage rice diseases early using cutting-edge deep learning techniques.

  ### Our Mission
    Rice is a staple food for millions, and its health is crucial for food security and farmer livelihoods. Our mission is to provide farmers with an intelligent, easy-to-use tool for early detection of rice diseases, ensuring timely intervention and improved crop yield.

  ### How It Works
    Our platform leverages MobileNet, a deep learning-based Convolutional Neural Network (CNN), to accurately identify four major rice diseases:

    - 🌿 Bacterial Blight – A bacterial infection causing leaf drying and wilting.
    - 🔥 Blast – A fungal disease that leads to lesions and affects crop productivity.
    - 🌑 Brown Spot – A fungal infection resulting in brown lesions on leaves.
    - 🦠 Tungro – A viral disease that causes yellowing and stunted plant growth.

    By simply uploading an image of the affected plant, our model analyzes and classifies the disease, providing farmers with real-time insights for effective disease management.
             
  ### 🔬 Technology Behind the Platform:  
  ------------------------------------------
    - **Deep Learning Models** – We use **CNNs and Transfer Learning** for high-precision detection.  
    - **Cloud Deployment** – Our AI model runs in the cloud for **real-time predictions**.  
    - **User-Friendly Interface** – A seamless web app designed for ease of use.  

   ##### 👨‍💻 Developed By: This project is built by **Mahaveer**, a passionate AI researcher specializing in **deep learning applications domain**. 
    

  """)
  image_path_graph ="rice-mobile-net-graph.png"
  st.image(image_path_graph,use_container_width =True)
  image_path_matrix ="rice-mobile-net-accurary-matrix.png"
  st.image(image_path_matrix,use_container_width =True)

elif(app_page_select=="Contact"):
  st.header("Contact US - Page")
  st.markdown("""
    --------------------------------------------
    We’d love to hear from you! Whether you have a question, need assistance, or just want to say hello, feel free to reach out to us.

    
    ### Get in Touch  
    -----------------------
      - 📍 Address: [Your Business Address]
      - 📞 Phone: [Your Contact Number]
      - 📧 Email: [Your Email Address]
    
    ### Business Hours
    --------------------
      - 🕒 Monday – Friday: 9:00 AM – 6:00 PM
      - 🕒 Saturday: 10:00 AM – 4:00 PM
      - ❌ Sunday: Closed
              
    ### Follow Us
    --------------------
        Stay connected and follow us on social media for updates, news, and special offers.

        🔗 [Facebook] | [Instagram] | [Twitter] | [LinkedIn]

        We look forward to assisting you!
  """)
