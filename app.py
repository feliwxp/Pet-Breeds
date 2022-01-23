from fastai.vision.all import *
import streamlit as st

model = load_learner('export.pkl')

st.write("""
         # Pet Breed Prediction
         """
         )
st.write("This is a simple image classification web app to predict the breed of the pet image uploaded.")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    image1 = PILImage.create(file)
    
    pred,pred_idx,probs = model.predict(image1)
    prob = np.array2string(probs[pred_idx].numpy())
    
    st.write("This is a " + pred + ". Accuracy: " + prob)
