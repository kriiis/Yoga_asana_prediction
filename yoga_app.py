import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow.keras import preprocessing
from tensorflow.keras import models
import tensorflow_hub as hub
# ----Global settings ------
yoga_img = Image.open('yoga-exercise-silhouette-3.jpg')

st.set_page_config(page_title="Yoga Asana Prediction", page_icon=yoga_img, layout="centered", initial_sidebar_state="auto", menu_items=None)

st.set_option('deprecation.showfileUploaderEncoding', False)

s = f"""<style> div.stButton > button:first-child {{background-color: #eeeeee;}} .sidebar .sidebar-content {{background-color: "#f55e61";}} <style>"""

st.markdown(s, unsafe_allow_html=True)

# ----- SIDEBAR ------
sil1_img = Image.open('goddess-silhouette.jpg')
sil2_img = Image.open('treesilhouette.png')
sil3_img = Image.open('yoga-physical-fitness-plank_silhouette.jpg')
sil4_img = Image.open('downward-dog.png')
sil5_img = Image.open('warrior2-silhouette.png')

sil_asanas1, mid, sil_asanas2 = st.sidebar.columns([1,1,1.4])

with sil_asanas1:
    st.image(sil1_img)
with sil_asanas2:
    st.image(sil2_img)

sil_asanas3, sil_asanas4, sil_asanas5 = st.sidebar.columns([1,1,1])
with sil_asanas3:
    st.image(sil3_img)
with sil_asanas4:
    st.image(sil4_img)
with sil_asanas5:
    st.image(sil5_img)


st.sidebar.header('Upload your Yoga Pose Image')
file_uploaded = st.sidebar.file_uploader('',type = ['jpg','jpeg','png'])

st.sidebar.header('Predict the Yoga Asana')
pred_asana = st.sidebar.button('Predict Yoga Asana')

st.sidebar.header('Plot Confusion Matrix')
cm_plot = st.sidebar.button('Plot Matrix') 
    

# ---- Main Output -----



col1, mid, col2 = st.columns([2,1,20])
with col1:
    st.image(yoga_img, width=80)
with col2:
    st.title('Yoga Asana Classification')
 
st.write("""
When practising yoga it is important to correctly train the yoga asanas.\n
This App helps you to identify a yoga asana and gives you detailed information on the correct execution of the asana. \n
So far the prediction model is trained on 5 yoga asanas: **downward facing dog, goddess pose, plank pose, tree pose and warrior 2** and has an accuracy of **91%**. \n
""")
    
classes = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']
all_preds = ['0.2','0.2','0.2','0.2','0.2']



def predict_class(image):
    classify_model = tf.keras.models.load_model('models/data_aug_model')
    test_image = image.resize((150,150))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255.
    prediction = classify_model.predict(test_image)
    winner = np.argmax(prediction)
    all_preds = prediction[0]
    predicted_class = f"This Yoga asana is predicted as {classes[winner]}"
    return predicted_class, all_preds, winner

def asana_description(winner):
    
    if winner == 0:
        text = st.markdown(""" <h4>Downward-Facing Dog</br><i>(Adho Mukha Svanasana)</i></h4>
        <ul>
        <li>Protect your elbows from hyperextension by pressing your inner arms away from each other</li>
        <li>Bend your knees to prevent rounding your back</li>
        <li>Also keep your heels away from the ground to straighten your back</li>
        <li>Push the ground away with your hands and draw your shoulders towrads your tailbone</li>
        <li>Keep your head bettween your upper arms to have a long spine and healthy neck</li></ul>""", True)
    elif winner == 1:
        text = st.markdown(""" <h4>Goddess Pose</br><i>(Utkata Konasana)</i></h4>
        <ul>
        <li>Keep your knees over your ankles and turn your feet out about 45 degrees</li>
        <li>Relax your shoulders and drop them down</li>
        <li>Keep your arms aktive as holding a big ball over your head or hold your hands in Anjali Mudra in front of your chest</li>
        <li>Engage your core and lengthen your tailbone towards the floor</li>
        </ul>""",True)
    elif winner == 2:
        text = st.markdown(""" <h4>Plank Pose</br><i>(Phalakasana)</i></h4>
        <ul>
        <li>To avoid a hollow back draw the lower belly in</li>
        <li>Engage your thigh muscles and lengthen the tailbone towards your heels</li>
        <li>Press your hands into the floor and keep pushing the floor away, shoulders over wrists</li>
        <li>Avoid sinking into your shoulders, firm shoulder blades against your back and then spread them away from the spine </li>
        </ul>""",True)
    elif winner == 3:
        text = st.markdown(""" <h4>Tree Pose</br><i>(Vrikshasana)</i></h4>
        <ul>
        <li>Bring all your weight on your standing leg and press evenly into all 4 corners of the foot</li>
        <li>Place your foot on the calf or upper inner thight, but not on your knee joint</li>
        <li>Relay shoulders away from ears and draw shoulder blades toward each other for an open chest</li>
        <li>Activate core and level hips to really extend from base to fingertips</li>
        </ul>""",True)
    else:
        text = st.markdown(""" <h4>Warrior 2</br><i>(Virabhadrasana II)</i></h4>
        <ul>
        <li>Distribute weight evenly between both legs with front leg bent and directly stacked over the ankle</li>
        <li>Keep your front knee straight and do not dodge to left or right</li>
        <li>Back foot parallel to back of the mat and outer edge grounded</li>
        <li>Keep the shoulders stacked over your hips</li>
        <li>Arms enganged and parallel to the floor, draw shoulders away from neck</li>
        </ul>""",True)
    return text

def confusion_ma(y_true, y_pred, class_names):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred, normalize=None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    return plt.show()


if file_uploaded is not None:
    image = Image.open(file_uploaded)
else:
    st.warning('Please upload an Image to see the predicted yoga asana')
    
if pred_asana:   
    figure = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    result, all_preds, winner = predict_class(image)
    st.subheader(result)
    
    col_img, col_text = st.columns(2)
    
    with col_img:
        st.pyplot(figure) 
    
    with col_text:
        asana_description(winner)
        
    st.subheader('Predicted values for each of the five yoga asanas')
    col1, col2, col3, col4, col5 = st.columns(5)
    
    image1 = Image.open('data/test/downdog/205.jpg')
    image2 = Image.open('data/test/goddess/goddess_033.jpg')
    image3 = Image.open('data/test/plank/plank_005.jpg')
    image4 = Image.open('data/test/tree/207.jpg')
    image5 = Image.open('data/test/warrior2/warrior2_002.jpg')

    with col1:
        st.subheader(round(all_preds[0] * 100,2))
        st.image(image1, caption='Downdog')

    with col2:
        st.subheader(round(all_preds[1] * 100,2))
        st.image(image2, caption='Goddess')

    with col3:
        st.subheader(round(all_preds[2] * 100,2))
        st.image(image3, caption='Plank')   
    
    with col4:
        st.subheader(round(all_preds[3] * 100,2))
        st.image(image4, caption='Tree')    
    
    with col5:
        st.subheader(round(all_preds[4] * 100,2))
        st.image(image5, caption='Warrior 2')    

else:
    st.write('')
    
cm_image = Image.open('confusion_matrix.jpg')

if cm_plot:
    st.subheader('Confusion Matrix')
    st.write('The Confusion Matrix shows the overall performance of the Convolutional Neural Network in predicting the correct Asana from a Test Image. The model has been trained with 1147 images of 5 different Yoga Asanas. The Model has been evaluated with 156 Test images. The correctly predicted Asanas can be seen in the diagonal of the confusion matrix, where the predicted and the true labels are the same. Also the wrongly predicted asanas can be seen where the predicted label is not equal to the true label. The trained model predicts the yoga asanas with an accuracy of 91%.')
    st.image(cm_image, width = 600)
else:
    st.write() 
