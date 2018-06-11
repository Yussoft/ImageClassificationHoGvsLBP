# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 10:54:27 2018

@author: Yus
"""
import numpy as np
import cv2
import os 
from sklearn.svm import SVC
from sklearn.model_selection import  cross_validate
import LBP as lbp
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA

workspace = "C:\\Users\\Yus\\Desktop\\ECI.Practica\\ImageClassificationHoGvsLBP"
data_dir = workspace + "\\data"

# README:
# PARA CAMBIAR EL DESCRIPTOR UTILIZADO BUSCA: method = methods y selecciona 
# el descriptor deseado de la lista methods.
#
# PARA CAMBIAR EL KERNEL DE SVM BUSCA: svm_model = SVC(kernel= y cambia el kernel
# de "linear" a "rbf".
# 
# AL ELEGIR LBP DEBES ACTIVAR O DESACTIVAR LA VERSION UNIFORME Y PCA, BUSCA:
# "use_pca = " y "uniform = " y activa o desactiva a tu gusto.

# Boolean to turn on or off the debugging messages
message = True

def read_imgs(path, label, messages=True):
    current_dir = os.getcwd()
    os.chdir(path)
    files = os.listdir()
    labels = []
    
    imgs = []
    for file in files:
        imgs.append(cv2.imread(file,0))
        labels.append(label)
    
    if messages: print("Path:",path,"\nImagenes leidas:",len(imgs))
    # Set old dir
    os.chdir(current_dir)
    return imgs, labels   

def hog_descriptor(img):
    hog = cv2.HOGDescriptor()
    return(hog.compute(img))

def compute_hog_set(dataset):
    computed_set = []
    for img in dataset:
        computed_set.append(hog_descriptor(img).flatten())
    return(computed_set)

def show_img(img, name="Image"):
    cv2.imread(name, img)
    cv2.waitKey(0)
    
###############################################################################
# LABELS: BACKGROUND 0 PEDESTRIANS 1
ejecucion = "hog linear"
print(ejecucion)
# Read images from train and test folders
if message: print("LOADING TRAIN/TEST IMAGES...")
train_backgrounds, train_back_labels = read_imgs(data_dir+"\\train\\background", 0)
train_pedestrians, train_ped_labels = read_imgs(data_dir+"\\train\\pedestrians", 1)
test_backgrounds, test_back_labels = read_imgs(data_dir+"\\test\\background", 0)
test_pedestrians, test_ped_labels = read_imgs(data_dir+"\\test\\pedestrians", 1)

# Complete set X,y
X = train_backgrounds + train_pedestrians + test_backgrounds + test_pedestrians
y = train_back_labels + train_ped_labels + test_back_labels + test_ped_labels

if message: print("IMAGES LOADED.\n")

methods = ["HOG","LBP","HOGLBP"]
method = methods[0]

if method == "HOG":
    print("__________________________________________________________________")
    print("Computing HOG...")
    X_train = compute_hog_set(X)
    
elif method == "LBP":
    print("__________________________________________________________________")
    print("Computing LBP...")
    # Calculate LBP values for each image
    lbp_images = []
    for i in range(0, len(X)):
        lbp_images.append(lbp.lbp_compute(X[i]))
        
    # Calculate histograrms for each image
    uniform = False
    print("Computing Histograms...uniform:",str(uniform))
    
    histograms = []
    for i in range(0, len(lbp_images)):
        # IMPORTANT, set uniform to False or True for LBP/LBPU
        histograms.append(lbp.lbp_hist(lbp_images[i], step=8, win_size=16, 
                                       uniform = uniform))
    X = [elem for histogram in histograms for elem in histogram]


    for i in range(0,len(histograms)):
        histograms[i] = np.concatenate(histograms[i])
    
    use_pca = False
    
    if use_pca: 
        print("PCA:",str(use_pca))
        # Due to high dimension, use PCA
        pca = PCA(random_state=77183)
        pca.fit(histograms)
        X_train = pca.components_
    else:
        X_train = histograms
    
elif method == "HOGLBP":
    print("COMPUTING HOG-LBPU")
    print("__________________________________________________________________")
    print("Computing HOG...")
    # Compute HOG
    hog = compute_hog_set(X)
    
    print("__________________________________________________________________")
    print("Computing LBP...")
    # Compute LBPU
    lbp_images = []
    for i in range(0, len(X)):
        lbp_images.append(lbp.lbp_compute(X[i]))
        
    # Calculate histograrms for each image
    print("Computing Histograms...")
    histograms = []
    for i in range(0, len(lbp_images)):
        # IMPORTANT, set uniform to False or True for LBP/LBPU
        histograms.append(lbp.lbp_hist(lbp_images[i], step=8, win_size=16, uniform = True))
    X = [elem for histogram in histograms for elem in histogram]

    for i in range(0,len(histograms)):
        histograms[i] = np.concatenate(histograms[i])
    
    # Concatenate both descriptors and apply PCA
    hoglbp = []
    for i in range(0, len(X)):
        hoglbp = np.concatenate(hog[i],histograms[i])
        
    # Due to high dimension, use PCA
    pca = PCA(random_state=77183)
    pca.fit(hoglbp)
    X_train = pca.components_

# Model creation and evaluation
scoring = {'accuracy': make_scorer(accuracy_score),
           'prec': 'precision',
           'recall': 'recall'}

print("__________________________________________________________________")
print("Evaluating the model with KFoldCV...")
svm_model = SVC(kernel="linear")

cv_results = cross_validate(svm_model, X_train, y, 
                            return_train_score=False, cv = 5, 
                            scoring=scoring)
print("Finished.")



