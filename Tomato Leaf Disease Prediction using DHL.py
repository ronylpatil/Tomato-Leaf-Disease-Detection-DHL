#!/usr/bin/env python
__author__ = "Ronil Patil"
__license__ = "Feel free to copy."

"""
Created on Fri Dec 3 09:37:46 2021

Project Topic : Tomato Leaf Disease Prediction using Deep Hybrid Learning
               
@author : Ronil Patil
Dataset from : https://www.kaggle.com/arjuntejaswi/plant-village
"""

# lets first use transfer learning to classify tomato leaf disease. There are 5 categories of disease 
# including healthy leafs. Here we are going to use VGG16 pretrained model. So let's begin...

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import glob
import seaborn as sns
import os
import random
import cv2
from sklearn import preprocessing
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import xgboost as xgb

# Creating static and local variables
SIZE = 256
SEED_TRAINING = 121
SEED_TESTING = 197
SEED_VALIDATION = 164
CHANNELS = 3
n_classes = 5
EPOCHS = 50
BATCH_SIZE = 16
input_shape = (SIZE, SIZE, CHANNELS)

#--------------------Training
def training(path) : 
    # loading data from local directory -> basic method. here labels are folder name, means each variety of data stored in particular folder.
    train_images = []       # training dataset stored here...(numpy array form of images)
    train_labels = []     # labels will be stored here 
    
    '''here we are using glob for accessing directories'''
    path = path + '\*'
    for directory_path in glob.glob(path) :   
        label = directory_path.split('\\')[-1]       # taking labels from folders
        # print(label)    # extracting label from directory path
        
        '''now we are entering into each folder and reading images from it and at a same 
        time we are also storing the label.'''
        for img_path in glob.glob(os.path.join(directory_path, '*.JPG')) :    
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)    # read color image 
            img = cv2.resize(img, (SIZE, SIZE))        # resize the image
            
            '''actually cv2 read image in BGR channel ordering, in color image we have 3 channels
            RGB so here the channel order is different nothing special!. it doesnt affect on model.
            In reality we can arrange them in any order we like.'''
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            train_images.append(img)
            train_labels.append(label)
    
    # Shuffling the list to avoid the some kind of bias.
    train_data = list(zip(train_images, train_labels))
    '''Seed function is used to save the state of a random function, so that it can generate          
        same random numbers on multiple executions of the code on the same machine or on 
        different machines (for a specific seed value).'''
    random.seed(SEED_TRAINING)   
    random.shuffle(train_data)
    train_images, train_labels = zip(*train_data)   # it will unzip the ziped iterators, it will return tuple
    
    # converting tuples to numpy array.
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    # let's normalize our pixel values 
    train_images = train_images / 255.0
    return train_images, train_labels

def testing(path) : 
    test_images = []
    test_labels = []
    
    path = path + '\*'
    for directory_path in glob.glob(path) : 
        labels = directory_path.split('\\')[-1]
        for img_path in glob.glob(os.path.join(directory_path, '*.JPG')) : 
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SIZE, SIZE))
            test_images.append(img)
            test_labels.append(labels)
            
    # Shuffling testing data
    test_data = list(zip(test_images, test_labels))
    random.seed(SEED_TESTING)
    random.shuffle(test_data)
    test_images, test_labels = zip(*test_data)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    # let's normalize our pixel values
    test_images = test_images / 255.0
    return test_images, test_labels

# preprocessing training and testing images
X_test, y_test_labels = training(r'E:\Tomato Leaf Disease - DHL\Tomato Leaf Dataset\Test')
X_train, y_train_labels = training(r'E:\Tomato Leaf Disease - DHL\Tomato Leaf Dataset\Train')

# encoding labels from text to integer
le = preprocessing.LabelEncoder()
le.fit(y_train_labels)
train_label_encoded = le.transform(y_train_labels)
le.fit(y_test_labels)
test_label_encoded = le.transform(y_test_labels)

# extracting original labels, later we will need it.
labels = dict(zip(le.classes_,range(len(le.classes_))))
print(labels)

# aliasing for better understanding
y_train, y_test = train_label_encoded, test_label_encoded

# let's load VGG16 Architecture without fully connected layers, considerding only fully convolutional layers
vgg_model = VGG16(weights = 'imagenet',  include_top = False, input_shape = (SIZE, SIZE, 3)) 

# let's make all layers non-trainable
for layer in vgg_model.layers : 
    layer.trainable = False
    
# now trainable parameter will be 0 in our architecture
vgg_model.summary()

# let's extract features from convolutional network for XBG
feature_extractor = vgg_model.predict(X_train)

# actually our data in the form of (2500, 8, 8, 512) into (2500, 8*8*512) 
features = feature_extractor.reshape(feature_extractor.shape[0], -1)
X_train_features = features

# perform same operation on test dataset
feature_extractor_test = vgg_model.predict(X_test)
features_test = feature_extractor_test.reshape(feature_extractor_test.shape[0], -1)
X_test_features = features_test

# defining Random Forest Classifier Model
rfc = RandomForestClassifier() 
rfc.fit(X_train_features, y_train)
rfc_pred = rfc.predict(X_test_features)

# inversing le transforme to get original labels
rfc_pred = le.inverse_transform(rfc_pred)

# let's check overall accuracy
print('Accuracy of Random Forest : ', metrics.accuracy_score(y_test_labels, rfc_pred))
'''
Accuracy of Random Forest :  0.83
'''

# defining XGBoost Classifier model      ----- Model trained in just 7 minutes(CPU)
model = xgb.XGBClassifier(use_label_encoder = False)
model.fit(X_train_features, y_train)
xgb_pred = model.predict(X_test_features)

# inversing le transforme to get original labels
xgb_pred = le.inverse_transform(xgb_pred)

# let's check overall accuracy
print('Accuracy of XGBOOST : ', metrics.accuracy_score(y_test_labels, xgb_pred))
'''
Accuracy of XGBOOST :  0.897
'''

# Confusion Matrics : Verify accuracy of each class
cm = confusion_matrix(y_test_labels, rfc_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm, annot = True).set_title('Random Forest Preformance')

cm = confusion_matrix(y_test_labels, xgb_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm, annot = True).set_title('XGB Preformance')

# Classification report
print('Random Forest Report : ')
print(classification_report(y_test_labels, rfc_pred))

print('XGB Report : ')
print(classification_report(y_test_labels, xgb_pred))
