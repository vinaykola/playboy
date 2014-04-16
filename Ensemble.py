# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from __future__ import division
import numpy as np
from scipy.sparse import *

# <codecell>

def ensemble(xtrain, ytrain, xtest, classifier, classifier_count):
    clf = AdaBoostClassifier(classifier,n_estimators=classifier_count)
    clf.fit(xtrain,ytrain)
    yout = clf.predict(xtest)
    return yout

# <codecell>

def readData():
    DATA_DIR = 'dir_data/'
    TEST_DIR = DATA_DIR + 'test/'
    TRAIN_DIR = DATA_DIR + 'train/'
    VALIDATION_DIR = DATA_DIR + 'validation/'
    
    d_dir = []
    d_dir.append(TRAIN_DIR)
    d_dir.append(TEST_DIR)
    
    features = {}
    features['audio'] = ['audio_' + x for x in ['mfcc','sai_boxes','sai_intervalgrams','spectrogram_stream','volume_stream']]
    features['text'] = ['text_' + x for x in ['description_unigrams','game_lda_1000','tag_unigrams']]
    features['vision'] = ['vision_' + x for x in ['cuboids_histogram','hist_motion_estimate','hog_features','hs_hist_stream','misc']]
    
    file_names = []
    
    # Get file names for all train and test files
    
    for dir_type in d_dir:
        for k1,f_family in features.iteritems():
            for f_type in f_family:
                file_names.append(dir_type + f_type + '.txt')
    
# Get number of features for all files
    
    records = []
    
    for file_name in file_names:
        with open(file_name,'r') as f:
            cols = 0
            for i,line in enumerate(f):
               if i % 2 == 0:
                   continue
               else:
                    fs = line.split(' ')
                    for feature in fs[1:]:
                        k,v = feature.split(':')
                        if int(k) > cols:
                            cols = int(k) 
            print file_name
            print cols
            records.append(cols)
            
    
    
#create numpy arrays for each file
    
    sizes = [11931,97935]
    d_type = ['test','train']
    data = {}
    r = 0
    for i,d in enumerate(d_type):
        for k,f_family in features.iteritems():
            fet = {}
            for f_type in f_family: 
                fet[f_type] = np.zeros((sizes[i],records[r]))
                r += 1
        data[d] = fet
    
#Read data from each file
    
    for k1,dt in data.iteritems():
        for k2,feature in dt.iteritems():
            f_name = DATA_DIR + k1 +'/' + k2 + '.txt'
            with open(f_name,'r') as f:
                for i,line in enumerate(f):
                    if i % 2 == 0:
                        try:
                            _,id = line.split('\t')
                            id = int(id)
#                             print id
                        except ValueError:
                            break                
                    else:
                        fs = line.split(' ')
                        video_class = fs[0]
                        for feature in fs[1:]:
                            k,v = feature.split(':')
                            k = int(k) - 1
                            v = float(v)
                            if k1 == 'test':
                                feature[id-300001,k] = v
                            else:
                                feature[id-1,k] = v

# <codecell>

readData()

