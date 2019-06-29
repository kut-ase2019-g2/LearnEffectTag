from clarifai.rest import ClarifaiApp
import pandas as pd
import os
import sys
import glob
from tqdm import tqdm

api_key='key'
modelname = 'cui_test'
effect_tag = ['KiraKira', 'Heart', 'Notes', 'Fire', 'Central line', 'Rain',
            'Thunder', 'Lens flare', 'Comicalize', 'Blur'] #10まで

# 学習用画像と解答ラベル
image_list = ['images.jpg', 'images2.jpg']
ans_list = [['KiraKira'], ['KiraKira']]

# API setup
app=ClarifaiApp(api_key=api_key)

# model make or load
search_model = app.models.search(modelname)
if modelname != None:
    model = app.models.get(modelname)
else:
    model = app.models.create(modelname, concepts=effect_tag)

# image uploaded
for i in range(len(image_list)):
    app.inputs.create_image_from_filename(filename=image_list[i], concepts=ans_list[i])

# model train
model.train()
