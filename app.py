import numpy as np
from PIL import Image
import detection.detect as detect
import classification.classify as classify
import segmentation.segment as segment
import streamlit as st
import io
import cv2 as cv



def train_models():
    detect.train()
    print("[INFO] Training Detection model done!")
    classify.train()
    print("[INFO] Training Classification model done!")
    
    # RUN THE FOLLOWING FOR PREPERING INPUT DATA FOR TRAINIG SEGMENTATION MODEL 
    segment.prepare_input()    
    segment.train()
    print("[INFO] Training Segmentation model done!")

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo
    
    
    
def main():
    my_logo = add_logo(logo_path="DEMO_IMAGES/ihene.png", width=150, height=150)
    st.sidebar.image(my_logo)
    st.sidebar.markdown(
        """
        <div style="color: dark green; font-size: 30px; font-weight: bold; margin-top: -20px;">
            Keeza AI Models
        </div>
        """,
        unsafe_allow_html=True
    )
    #st.sidebar.subheader("Keeza AI Models")
    st.markdown(
    """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:300px;
            margin-left:-300px;
        }
        </style>
    """,
    unsafe_allow_html=True,
    )
    
    app_mode = st.sidebar.selectbox('Choose the App Mode', ['Select model', 'Object Detection LIVER MASS', 'Object Detection BLOOD CELL', "Object Segmentation BREAST MASS"])

    
    if app_mode == 'Select model':
        
        #st.header("Keeza AI models")
        st.warning("Detection: Liver Mass | Blood Cell")
        st.warning("Segmentation: Breast Mass")
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<hr style='border: 1px solid green'>", unsafe_allow_html=True)
            st.image('DEMO_IMAGES/liver_m2.png', caption='Liver US image', use_container_width=True)

        with col2:
            st.markdown("<hr style='border: 1px solid green'>", unsafe_allow_html=True)
            st.image('DEMO_IMAGES/liver_m1.png', caption='Segmentation', use_container_width=True)


    elif app_mode == "Object Detection LIVER MASS":
        st.header("Object Detection models",)
        st.markdown("<small> Liver Mass from ultrasound (US) images: Benign | Malignant <small>",unsafe_allow_html=True )
        
        st.sidebar.markdown("----")
        confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.35)
        
        img_file_buffer_detect = st.sidebar.file_uploader("Upload US Liver Images  ONLY", type=['jpg','jpeg', 'png'], key=0)
        DEMO_IMAGE = "DEMO_IMAGES/liver_mass_benign_1.jpg"
        
        if img_file_buffer_detect is not None:
            img_buffer = img_file_buffer_detect.read()
            img = np.array(Image.open(io.BytesIO(img_buffer)))  # Convert to NumPy array directly
            image = Image.open(io.BytesIO(img_buffer))  
        else:
            #img = cv.imread(DEMO_IMAGE)
            #image = np.array(Image.open(DEMO_IMAGE))
            img = np.array(Image.open(DEMO_IMAGE))  # Convert demo image to a NumPy array
            image = Image.open(DEMO_IMAGE) 
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        # predict
        detect.predict_liver_mass(img, confidence, st)

    elif app_mode == "Object Detection BLOOD CELL":
        
        st.header("Object Detection models",)
        st.markdown("<small> BLOOD CELL: WBC (white blood cells), RBC(red blood cells), and Platelets <small>",unsafe_allow_html=True )
        
        st.sidebar.markdown("----")
        confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.35)
        
        img_file_buffer_detect = st.sidebar.file_uploader("Upload Blood CELL Images ONLY", type=['jpg','jpeg', 'png'], key=0)
        DEMO_IMAGE = "DEMO_IMAGES/blood_cell.jpg"
        
        if img_file_buffer_detect is not None:
            # img = cv.imdecode(np.fromstring(img_file_buffer_detect.read(), np.uint8), 1)
            # image = np.array(Image.open(img_file_buffer_detect))
            img_buffer = img_file_buffer_detect.read()
            img = np.array(Image.open(io.BytesIO(img_buffer)))  # Convert to NumPy array
            image = Image.open(io.BytesIO(img_buffer)) 
        else:
            # img = cv.imread(DEMO_IMAGE)
            # image = np.array(Image.open(DEMO_IMAGE))
            img = np.array(Image.open(DEMO_IMAGE))  # Convert demo image to NumPy array
            image = Image.open(DEMO_IMAGE) 
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        # predict
        detect.predict(img, confidence, st)
        
    # elif app_mode == "Object Classification":
        
    #     st.header("Object Classification")
        
    #     st.sidebar.markdown("----")
        
    #     img_file_buffer_classify = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=1)
    #     DEMO_IMAGE = "DEMO_IMAGES/094.png"
        
    #     if img_file_buffer_classify is not None:
    #         img = cv.imdecode(np.fromstring(img_file_buffer_classify.read(), np.uint8), 1)
    #         image = np.array(Image.open(img_file_buffer_classify))
    #     else:
    #         img = cv.imread(DEMO_IMAGE)
    #         image = np.array(Image.open(DEMO_IMAGE))
    #     st.sidebar.text("Original Image")
    #     st.sidebar.image(image)
        
    #     # predict
    #     classify.predict(img, st)
        
    elif app_mode == "Object Segmentation BREAST MASS":
        
        
        st.header("Segmentation models")
        st.markdown("<small> Breast Mass from ultrasound (US) images: Benign | Malignant <small>", unsafe_allow_html=True)
        
        st.sidebar.markdown("----")
        
        img_file_buffer_segment = st.sidebar.file_uploader("Upload Breast US Images ONLY", type=['jpg','jpeg', 'png'], key=2)
        DEMO_IMAGE = "DEMO_IMAGES/breast_benign_10.png"
        
        if img_file_buffer_segment is not None:
            # img = cv.imdecode(np.fromstring(img_file_buffer_segment.read(), np.uint8), 1)
            # image = np.array(Image.open(img_file_buffer_segment))

            img_buffer = img_file_buffer_segment.read()
            img = np.array(Image.open(io.BytesIO(img_buffer)))  # Convert to NumPy array from Pillow
            image = Image.open(io.BytesIO(img_buffer))  
        else:
            # img = cv.imread(DEMO_IMAGE)
            # image = np.array(Image.open(DEMO_IMAGE))
            img = np.array(Image.open(DEMO_IMAGE))  # Convert demo image to NumPy array
            image = Image.open(DEMO_IMAGE)  # Keep
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        
        # predict
        segment.predict(img, st)

    st.markdown(
        """
        <script>
        setInterval(function() {
            fetch('/stream');
        }, 60000);  // Ping the server every minute
        </script>
        """,
        unsafe_allow_html=True
    )

        
    st.markdown("<hr style='border: 1px solid red'>", unsafe_allow_html=True)
    st.markdown("""<small>
    Keeza ~ Tech | +250788384528, +250788317992,+250785540835 | keey08@gmail.com | KN 78 St Norrsken Kigali Rwanda | 
    <small>""", unsafe_allow_html=True)

        
        
       
        


if __name__ == "__main__":
    try:
        
        # RUN THE FOLLOWING ONLY IF YOU WANT TO TRAIN MODEL AGAIN 
        # train_models()
        
        main()
    except SystemExit:
        pass
        

