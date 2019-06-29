from clarifai.rest import ClarifaiApp
import pandas as pd
import os
import sys
import glob
from tqdm import tqdm

api_key='key'
max_concepts=200
min_value=0.0
image_list = ['images.jpg', 'images2.jpg']
ans_list = ['KiraKira', 'KiraKira']
effect_tag = ['KiraKira', 'Heart', 'Notes', 'Fire', 'Central line', 'Rain',
            'Thunder', 'Lens flare', 'Comicalize', 'Blur', 'Change contrast',
             'Zoom in', 'Zoom out', ] #13
csv_file = 'effect_weight.csv'
zeros = [[0 for i in range(len(effect_tag))]]

# APIsetup
app=ClarifaiApp(api_key=api_key)
# model = app.public_models.general_model
model = app.models.get('learn_test')

# # READ csv
# if os.path.exists(csv_file):
#     df = pd.read_csv(csv_file, index_col=0)
# else:
#     df = pd.DataFrame(data=zeros,index=['COUNT'],columns=effect_tag)

# API do
# pbar = tqdm(total=int(len(image_list)), desc='Learn process')
for i in range(len(image_list)):
    response = model.predict_by_filename(image_list[i], max_concepts=max_concepts)
    rList = response['outputs'][0]['data']['concepts']
    for j in range(len(rList)):
        concept_name = str(rList[j]['name'])
        concept_weight = rList[j]['value']
        print(concept_name + ': ' +str(concept_weight))
        # if concept_name not in df.index.values:
    #         df.loc[concept_name] = 0
    #     df.loc[concept_name, ans_list[i]] = df.loc[concept_name, ans_list[i]] + concept_weight
    # df.loc['COUNT', ans_list[i]] = df.loc['COUNT', ans_list[i]] + 1
#     pbar.update(1)  # プロセスバーを進行
# pbar.close()  # プロセスバーの終了

# Save
# df.to_csv(csv_file, sep=",")
