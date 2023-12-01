from PIL import Image
import torch
import requests
from io import BytesIO
import pandas as pd
from torchvision import transforms as T

def preprocess_image(image):
    # Преобразование изображения в формат RGB
    transform = T.Compose([
        T.Resize((256, 256)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def preprocess_brains(image):
    image = image.convert('RGB')
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)


def transforming(image):
    transform = T.Compose([T.Resize((256, 256))])
    return transform(image)




def As_File(uploaded_image, model, decode):
    model.eval()
    image = Image.open(uploaded_image).convert("RGB")
    image_tensor = preprocess_image(image).to('cpu')
    with torch.no_grad():
        outputs = model(image_tensor)

    # Преобразование предсказаний в вероятности с использованием Softmax
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Получение индекса предсказанного класса
    _, predicted_class = torch.max(outputs.data, 1)
    predicted_class_index = int(predicted_class.item())
    res = f'Predicted Class: {decode.get(predicted_class_index, f"Class {predicted_class_index}")} с уверенностью: {round((probabilities.max() * 100).item(), 2)} %'
    return res


def ByURL(url, model, decode):
    model.eval()
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    image_tensor = preprocess_image(img).to('cpu')
    with torch.no_grad():
        outputs = model(image_tensor)

    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Получение индекса предсказанного класса
    _, predicted_class = torch.max(outputs.data, 1)
    predicted_class_index = int(predicted_class.item())
    res = f'Predicted Class: {decode.get(predicted_class_index, f"Class {predicted_class_index}")} с уверенностью: {round((probabilities.max() * 100).item(), 2)} %'
    return res
def show_file(img):
    return(img)

def check_url(url, restype):
    df = pd.read_csv('data.txt', names=['url', 'Class', 'Confidence', 'Model', 'Feedback'])
    if (url in df['url'].values) and (restype in df[df['url']==url]['Model'].values):
        return False
    else:
        return True

def detect_objects(image, model):
    # Преобразование изображения, чтобы соответствовать ожиданиям модели
    transform = T.Compose([
        T.Resize((640, 640)),  # Размер может отличаться в зависимости от вашей модели
        T.ToTensor(),
        # Другие преобразования, которые могут потребоваться
    ])

    input_image = transform(image).unsqueeze(0)  # Добавляем размерность батча

    # Переводим модель в режим оценки и применяем предсказание
    model.eval()
    with torch.no_grad():
        predictions = model(input_image)



    return predictions