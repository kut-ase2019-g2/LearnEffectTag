import csv, random
import glob
import sys
from clarifai.rest import ClarifaiApp

# 乱数シード値
random.seed(1)

api_key='ないしょ'
effect_tag = ['KiraKira', 'Change_contrast', 'Lens_flare', 'Rain', 'Central_line',
    'Zoom_in', 'Heart', 'Notes', 'Zoom_out']
max_concepts=200
min_value=0.0
all_image_list = glob.glob('image/*.*')
image_list = random.sample(all_image_list, 10)

# fileの書き出し setup
f_AI = open('tag_data_AI.csv', 'w')
w_AI = csv.writer(f_AI, lineterminator='\n')
f_Rn = open('tag_data_Random.csv', 'w')
w_Rn = csv.writer(f_Rn, lineterminator='\n')

# API setup
app = ClarifaiApp(api_key=api_key)
model = app.models.get('AutoEffectTagSet')

# API DO
for i in range(len(image_list)):
    response = model.predict_by_filename(image_list[i], max_concepts=max_concepts)
    rList = response['outputs'][0]['data']['concepts']
    w_AI.writerow([str(image_list[i])] + [str(rList[0]['name'])])
    w_Rn.writerow([str(image_list[i])] + [random.sample(effect_tag, 1)[0]])
