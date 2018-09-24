import os
import json
import base64
from skimage.io import imread

img_path = 'data/test/test_img.jpg'

# Image as list input
img = imread(img_path)
img_list = img.tolist()
json_data = {'image': img_list}
out_path = 'data/test/test_json_list.json'
with open(out_path, 'w+') as f:
    json.dump(json_data, f)

# Image as base64 encoded string
with open(img_path, 'rb') as f:  # encode the actual jpeg bytes not the raw image values
    img_bytes = base64.b64encode(f.read())
json_data = {'image_bytes': {'b64': img_bytes.decode('ascii')}}
out_path = 'data/test/test_json_b64.json'
with open(out_path, 'w+') as f:
    json.dump(json_data, f)

# Image as URL (NOTE: you might need to change the image URL below)
img_url = 'gs://tf-keras-deploy-mlengine/data/test/test_img.jpg'
json_data = {'image_url': img_url}
out_path = 'data/test/test_json_url.json'
with open(out_path, 'w+') as f:
    json.dump(json_data, f)

