import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }
    .title {
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction {
        font-size: 20px;
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        border-radius: 5px;
    }
    .cat {
        background-color: #ffd6d6;
    }
    .dog {
        background-color: #d6e5ff;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    try:
        model = load_model('cats_dogs_mobilenetv2.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image for prediction
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

def predict_image(model, img_path):
    """
    Make a prediction on a single image
    """
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    confidence = prediction[0][0]
    
    if confidence < 0.5:
        predicted_class = "Cat"
        confidence = 1 - confidence
    else:
        predicted_class = "Dog"
    
    return predicted_class, confidence

def main():
    st.title("🐱 Cats vs Dogs Classifier 🐶")
    st.markdown("Upload an image of a cat or dog, and our AI model will predict which one it is!")
    
    # Load model
    model = load_my_model()
    if model is None:
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the image
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Make prediction
            try:
                predicted_class, confidence = predict_image(model, temp_path)
                
                # Display prediction
                st.subheader("Prediction Result")
                
                if predicted_class == "Cat":
                    st.markdown(f'<div class="prediction cat">🐱 This is a <b>{predicted_class}</b>!<br>Confidence: {confidence:.2%}</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction dog">🐶 This is a <b>{predicted_class}</b>!<br>Confidence: {confidence:.2%}</div>', 
                               unsafe_allow_html=True)
                
                # Show confidence meter
                st.progress(float(confidence))
                st.caption(f"Confidence: {confidence:.2%}")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
            
        # Remove temporary file
        os.remove(temp_path)

if __name__ == "__main__":
    main()
