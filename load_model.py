import numpy as np
from tensorflow import keras
from keras.models import load_model

def load_and_predict_image():
    image_size = (180, 180)
    print("------------------------------------------------------------------")
    print("|                NEURAL NETWORK OBJECT CLASSIFICATION            |")
    print("------------------------------------------------------------------")
    predict = input("path: ")  # input 
    model = load_model("ObjectsModel")

    img = keras.utils.load_img(
        predict, target_size=image_size
    )
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(predictions[0])
    print(f"object {100 * (1 - score):.2f}% object 1 and {100 * score:.2f}% object 2.")
