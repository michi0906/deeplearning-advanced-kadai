from django.shortcuts import render
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = VGG16(weights='imagenet')

def index(request):
    result = None
    if request.method == 'POST' and request.FILES.get('image'):
        img = request.FILES['image']
        path = 'temp.jpg'
        with open(path, 'wb') as f:
            for chunk in img.chunks():
                f.write(chunk)

        img_data = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img_data)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        result = decode_predictions(preds, top=5)[0]
        os.remove(path)

    return render(request, 'index.html', {'result': result})
