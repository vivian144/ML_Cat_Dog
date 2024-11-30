from django.shortcuts import render
from .ml_model import predict_image
import os

def predict_view(request):
    result = None
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        image_path = os.path.join('uploads', image.name)

        # Save uploaded image
        with open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        # Predict the class
        result = predict_image(image_path)

        # Delete the image after prediction
        os.remove(image_path)

    return render(request, 'predict.html', {'result': result})
