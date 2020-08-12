import streamlit as st
import pandas as pd
import numpy as np
import pickle

html = """
  <style>
    .st-ek{
        border: 1px solid #ff2b2b;
        padding: 9px;
    }
  </style>
"""
st.markdown(html, unsafe_allow_html=True)

st.title("Surface Roughness Prediction Model")
# Title
st.sidebar.title("Surface Roughness Prediction")

# Header
st.sidebar.subheader("This Web app helps you to predict the surface roughness from related machining parameters i.e Cutting Speed (m/s), Flank Wear (mm), Feed per Tooth (mm/tooth), and Depth of cut (mm). The core intent of this application is to help you select the optimal machining parameters required to minimize the effect of surface roughness.")

# Description
st.sidebar.info("Predictive modelling techniques such as Artificial Neural Networks, Linear Regression, and Boosting techniques were used, and the error between the experimental and predicted values was minimized to ensure a reliable model. Overfitting was prevented using Cross-validation and other techniques.")

# Show image for Neural Network Architecture
from PIL import Image
nn_architecture = Image.open("./images/visualized_model.jpg")
loss = Image.open("./images/loss.jpg")

if st.sidebar.checkbox("Show ANN Architecture/Hide"):
    st.sidebar.image(nn_architecture, width = 400, caption = "Neural Network Architecture")

if st.sidebar.checkbox("Show Training Loss performance (Mean squared error)"):
    st.sidebar.image(loss, width = 400, caption = "Training and Validation Loss")

# SelectBox
occupation = st.selectbox("Your area of Specialization", ["Mechanical Engineering","Materials Science","Data Science","Machine learning"])

st.info("Please enter the Values for each Machining parameter to make a prediction")

# st.subheader("Cutting Speed (m/s)")
cutting_speed = st.number_input("Cutting Speed (m/s)", value = 1.)

# st.subheader("Feed per tooth (mm/tooth)")
feed_per_tooth = st.number_input("Feed per tooth (mm/tooth)", value = 1.)

# st.subheader("Depth of Cut (mm)")
depth_of_cut = st.number_input("Depth of Cut (mm)", value = 1.)

# st.subheader("Flank Wear (mm)")
flank_wear = st.number_input("Flank Wear (mm)", value = 1.)


# SelectBox
model = st.selectbox("Select Model", ["Artificial Neural Networks (ANN) Regressor","Linear Regression","Support Vector Regression","XGBOOST Regressor"])


features = {
    "Cutting Speed (m/s)" : cutting_speed,
    "Feed per Tooth (mm/tooth)" : feed_per_tooth,
    "Depth of cut (mm)" : depth_of_cut,
    "Flank Wear (mm)" : flank_wear
}

data = pd.DataFrame(features, index = ["Values"])

if st.button("Click Here to Predict"):
    st.write(data)
    st.write(model)
    values = np.asarray([cutting_speed, feed_per_tooth, depth_of_cut, flank_wear]).reshape(1,-1)
    
    if model == "Artificial Neural Networks (ANN) Regressor":
        pipeline = pickle.load(open("./models/random_forest.pickle", "rb"))
        prediction = pipeline.predict(values)[0]
        st.success("The Predicted Surface roughness is :-" + str(prediction))

    elif model == "Linear Regression":
        pipeline = pickle.load(open("./models/linear_regression.pickle", "rb"))
        prediction = pipeline.predict(values)[0]
        st.success("The Predicted Surface roughness is :-" + str(prediction))

    elif model == "Support Vector Regression":
        pipeline = pickle.load(open("./models/support_vector_regression.pickle", "rb"))
        prediction = pipeline.predict(values)[0]
        st.success("The Predicted Surface roughness is :-" + str(prediction))
        
    elif model == "XGBOOST Regressor":
        pipeline = pickle.load(open("./models/tree.pickle", "rb"))
        prediction = pipeline.predict(values)[0]
        st.success("The Predicted Surface roughness is :-" + str(prediction))

    else:
        st.danger("Please select a Model for Prediction")
