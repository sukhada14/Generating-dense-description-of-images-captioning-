
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tensorflow as tf


# In[2]:


from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# In[3]:


tf.__version__


# In[4]:


tf.keras.__version__


# In[5]:


import coco


# In[6]:


import nltk


# In[ ]:


nltk.download()


# In[7]:


import json
import os


# In[8]:


data_dir = "D:\WCE\BTech-1\my\Project\Dataset\coco"

# Sub-directories for the training- and validation-sets.
train_dir = "D:\WCE\BTech-1\my\Project\Dataset\coco\train\train2014"
val_dir = "D:\WCE\BTech-1\my\Project\Dataset\coco\val\val2014"

# Base-URL for the data-sets on the internet.
data_url = "http://images.cocodataset.org/"


# In[9]:


# helper function to load the data

def _load_records(train=True):
    """
    Load the image-filenames and captions
    for either the training-set or the validation-set.
    """

    if train:
        # Training-set.
        filename = "captions_train2014.json"
    else:
        # Validation-set.
        filename = "captions_val2014.json"

    # Full path for the data-file.
    path = os.path.join(data_dir, "annotations", filename)

    # Load the file.
    with open(path, "r", encoding="utf-8") as file:
        data_raw = json.load(file)

    # Convenience variables.
    images = data_raw['images']
    annotations = data_raw['annotations']

    # The lookup-key is the image-id.
    records = dict()

    # Collect all the filenames for the images.
    for image in images:
        # Get the id and filename for this image.
        image_id = image['id']
        filename = image['file_name']

        # Initialize a new data-record.
        record = dict()

        # Set the image-filename in the data-record.
        record['filename'] = filename

        # Initialize an empty list of image-captions
        record['captions'] = list()

        # Save the record using the the image-id as the lookup-key.
        records[image_id] = record

    # Collect all the captions for the images.
    for ann in annotations:
        # Get the id and caption for an image.
        image_id = ann['image_id']
        caption = ann['caption']

        # Lookup the data-record for this image-id.
        record = records[image_id]

        # Append the current caption to the list of captions in the
        record['captions'].append(caption)

    # Convert the records-dict to a list of tuples.
    records_list = [(key, record['filename'], record['captions'])
                    for key, record in sorted(records.items())]

    # Convert the list of tuples to separate tuples with the data.
    ids, filenames, captions = zip(*records_list)

    return ids, filenames, captions


# In[10]:


#loading training data
id, filenm, captions = _load_records(train_dir)


# In[11]:


#combining in one dataframe
import pandas as pd
input_data = {'id': id, 'filename': filenm, 'captions' :captions}
input_data = pd.DataFrame(data=input_data)
input_data


# In[12]:


#image loading
def load_image(path, size=None):
    
    img = Image.open(path)

    # Resize image if desired.
    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)

    # Convert image to numpy array.
    img = np.array(img)

    # Scale image-pixels so they fall between 0.0 and 1.0
    img = img / 255.0

    return img


# In[13]:


#showing image
def show_image(index, train):
    
    if train:
        # Use an image from the training-set.
        dir = 'D:/WCE/BTech-1/my/Project/Dataset/coco/train/train2014'
        filename = input_data.filename[index]
        captions = input_data.captions[index]
    else:
        # Use an image from the validation-set.
        dir = 'D:/WCE/BTech-1/my/Project/Dataset/coco/val/val2014'
        filename = val_data.filename[index]
        captions = val_data.captions[index]

    # Path for the image-file.
    path = os.path.join(dir, filename)
    
    # Load the image and plot it.
    img = load_image(path)
    plt.imshow(img)
    plt.show()
    
     # Print the captions for this image.
    for caption in captions:
        print(caption)


# In[14]:


for i in range(0,10):
    show_image(index=input_data.index[i], train=True)


# In[15]:


caption_corpora=''
for i in range(0,len(input_data)):
    caption_corpora = caption_corpora.join(input_data.captions[i])

