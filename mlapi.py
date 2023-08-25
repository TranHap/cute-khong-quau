from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import requests

from keras.models import load_model

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",  # Add your frontend's URL here
    "https://freebie-kappa.vercel.app",
    "https://freebie-kappa.vercel.app/upload",
]
model = load_model("keras_model_2.h5", compile=False)
class_names = open("labels_6.txt", "r").readlines()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class imageURL(BaseModel):
    image_url: str 

async def ourImage(image_url: str):
    # async with httpx.AsyncClient() as client:
        # response = await client.get(image_url)
        # image_url = image_url.decode('utf-8')
        response = requests.get(image_url)
        image_data = response.content
        image = Image.open(BytesIO(image_data))
        image = np.array(image)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1
        print(image)
        return {"status_code": response.status_code, "image": image}

@app.post("/")
async def scoring_endpoint(item: imageURL):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    image_url = df.iloc[0,0]
    image = await ourImage(image_url)
    print(image["status_code"])
    image = image["image"]
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
   
    return {"label": str(class_name[2:])}