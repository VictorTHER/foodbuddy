import numpy as np
import pandas as pd
import os
import json
from matplotlib import pyplot as plt
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

client = storage.Client()

bucket_name = 'foodbuddy-dataset'
bucket = client.get_bucket(bucket_name)

def create_dataframe(annotations_path):
    with open(annotations_path, 'r') as file:
        data = json.load(file)

    images = pd.DataFrame(data['images']).rename(columns={'id': 'image_id'})[['image_id', 'file_name', 'width', 'height']]

    categories = pd.DataFrame(data['categories'])[['id', 'name', 'name_readable']]
    categories.rename(columns={'id': 'category_id'}, inplace=True)

    usecols = ['image_id', 'category_id']
    annotations = pd.DataFrame(data['annotations'])[usecols]

    dataframe = annotations.merge(categories, on='category_id').merge(images, on='image_id')[['file_name', 'name', 'name_readable']]

    return dataframe


annotation_train_path = './raw_data/public_training_set_release_2.0/annotations.json'
image_train_path = './raw_data/public_training_set_release_2.0/images'
annotation_val_path = './raw_data/public_validation_set_2.0/annotations.json'
image_val_path = './raw_data/public_validation_set_release_2.0/images'

train_df = create_dataframe(annotation_train_path).drop_duplicates().reset_index(drop=True)
val_df = create_dataframe(annotation_val_path).drop_duplicates().reset_index(drop=True)
