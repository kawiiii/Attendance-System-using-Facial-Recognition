import os
import numpy as np
from PIL import Image
import pickle
import cv2

path = os.path.dirname(os.path.abspath(__file__))
base_directory = os.path.join(path, "images")

recognizer = cv2.face.LBPHFaceRecognizer_create()

names = []
CMS_IDS = []
# accessing the name folders with training photos which are inside CMS ID folders
for file in os.listdir(base_directory):
    CMS_IDS.append(file)
    for name in os.listdir(os.path.join(base_directory, file)):
        names.append(name)

labels = []
training_data = []

for cms_id in CMS_IDS:
    inside_base_dir = os.path.join(cms_id, names[CMS_IDS.index(cms_id)])
    image_directory = os.path.join(base_directory, inside_base_dir)

    for image in os.listdir(image_directory):
        image_path = os.path.join(image_directory, image)
        image_ = Image.open(image_path)

        # converting image to np array
        image_array = np.array(image_, "uint8")
        current_id = CMS_IDS.index(cms_id)
        labels.append(current_id)
        training_data.append(image_array)

# converting to np arrays
labels = np.array(labels)
# training data
recognizer.train(training_data, labels)
# saving training data to a yml file
recognizer.save('trainer.yml')

# storing names in a file
with open("names.pickle", "wb") as file:
    pickle.dump(names, file)

# string cms ids in a file
with open("cms_ids.pickle", "wb") as file:
    pickle.dump(CMS_IDS, file)
print("Training complete")
