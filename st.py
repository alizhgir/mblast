import streamlit as st
from PIL import Image
import torch
import glob
from torchvision import transforms
from PIL import Image, ImageDraw

from prediction import preprocess_image, transforming, detect_objects
from our_models import ImprovedConvAutoencoder

model = torch.hub.load(
    'ultralytics/yolov5', # пути будем указывать гдето в локальном пространстве
    'custom', # непредобученная
    force_reload=True,
    path='best2.pt', # путь к нашим весам
    )


def infer_image(img):
    model.conf = confidence
    result =  model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


def image_input(data_src):
    img_file = None
    if data_src == 'Sample data':
        # get all sample images
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

class_mapping = {}


# Function to detect objects

def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Brain Checker", "Clearing", "Preprocessing"])

    if selected_page == "Brain Checker":
        page_home()
    elif selected_page == "Clearing":
        page_clearing()
    elif selected_page == "Preprocessing":
        page_preprocessing()
def page_home():

    global confidence, model

    st.title("Your astrological forecast based on tarot cards via brain MR")

        # load model

    model.eval()

        # confidence slider
    confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = infer_image(image)
        st.image(img, caption="Model prediction")




# Function to draw boxes on the image










def page_preprocessing():
    st.title('Preprocessing')





def page_clearing():
    st.title('Clear your docs')
    model_cleaning = ImprovedConvAutoencoder()
    model_cleaning.load_state_dict(torch.load('weights/improved_model_weights.pth', map_location=torch.device('cpu')))
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with torch.no_grad():
            clean_doc = model_cleaning(preprocess_image(image))
        col3, col4 = st.columns(2)
        with col3:
            st.image(transforming(image), caption='Before')

        with col4:
            st.image(transforms.ToPILImage()(clean_doc.squeeze(0)), caption='After')


if __name__ == "__main__":
    main()