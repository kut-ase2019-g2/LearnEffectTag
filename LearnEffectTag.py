from clarifai.rest import ClarifaiApp
import pandas as pd
import os
import sys
import glob
from tqdm import tqdm

api_key='ないしょ'
modelname = 'AutoEffectTagSet'
# effect_tag = ['KiraKira', 'Change_contrast', 'Lens_flare', 'Rain', 'Central_line',
#     'Zoom_in', 'Heart', 'Fire', 'Notes', 'Zoom_out'] #10まで
effect_tag = ['KiraKira', 'Change_contrast', 'Lens_flare', 'Rain', 'Central_line',
     'Zoom_in', 'Heart', 'Notes', 'Zoom_out'] #10まで
csvfile = 'SetTag.csv'
# 学習用画像と解答ラベル
# image_list = ['images.jpg', 'images2.jpg']
# ans_list = [['KiraKira'], ['KiraKira']]
df = pd.read_csv(csvfile, index_col=0)
# API setup
app=ClarifaiApp(api_key=api_key)

# model make or load
search_model = app.models.search(modelname)
if search_model != None:
    model = app.models.get(modelname)
else:
    model = app.models.create(modelname, concepts=effect_tag)

# image uploaded
pbar = tqdm(total=len(df), desc='uploaded')
for i in range(len(df)):
    filename = df.index[i]
    concepts = []
    for j in range(len(df.columns)):
        if df.iat[i,j] != 'None':
            concepts.append(df.iat[i,j])
    # print(filename + ': ')
    # print(concepts)
    app.inputs.create_image_from_filename(filename=filename, concepts=concepts)
    pbar.update(1)

# model train
print('train now!')
model.train()
print('train end!')
