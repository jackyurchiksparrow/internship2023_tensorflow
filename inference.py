from argparse import ArgumentParser
import os
import pandas as pd
import pathlib
from tensorflow import keras
import cv2
import numpy as np

#add CLI argument as a file path to samples directory
parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="samples_dir",
                    help="Directory with test samples", required=True) # it's required

args = parser.parse_args()
samples_dir = args.samples_dir

# if it's not directory - terminate
if os.path.isfile(samples_dir):
    print("it's a file!")
    exit()

# otherwise find all the images in the folder and append their POSIX path to content list
content = []
for file in os.listdir(samples_dir):
    if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
        content.append({'Posix_path':pathlib.PurePosixPath("{0}\{1}".format(samples_dir, file))})
    
# parse list to dataframe for better visualizing and structuring
images = pd.DataFrame(content)

# if there are no images - terminate
if images.empty:
    print("No images found")
    exit()

# load the pre-trained earlier model
model = keras.models.load_model('model_letters_numbers_vin.h5')

# read the mapping file provided by emnist to know which character is predicted
character = [0]*47 # 47 characters in the file

with open('emnist-balanced-mapping.txt') as f: 
    lines = f.readlines() 
    for line in lines:
        a, b = map(int, line.split())
        character[a] = chr(b)

character = np.array(character)

# list of future ASCII indexes
ascii_inxs = []

# for every image we find
for path in images['Posix_path'].values:
    img = cv2.imread(str(path)) # read it

    img = img[4:69, 8:65] # remove the white border
    img = cv2.resize(img, (28,28)) # resize to the suitable size
    img = np.invert(img) # invert black and white
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # turn image to grayscale
    # expanding dimnesions and normalizing
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr = img_arr.reshape((1, 28, 28, 1))
    # predicting
    result = model.predict([img_arr])
    ascii_inxs.append(ord(character[np.argmax(result[0])]))

images['ASCII_index'] = ascii_inxs

os.system('cls')
print(images)




