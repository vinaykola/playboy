from __future__ import division
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import cPickle as pickle
import numpy as np

import scipy.sparse


import os.path

# Checks for a pickle with the file name specified ands loads that variable
# If the file doesn't exist, computes the value using the function supplied and pickles it for next time

def load_variable(f_name,compute,*args):
    if os.path.isfile(f_name):
        with open(f_name,'rb') as f:
            a = pickle.load(f)
    else:
        a = compute(*args)
        with open(f_name,'wb') as f:
            pickle.dump(a,f)
    return a

def ensemble(xtrain, ytrain, xtest, classifier, classifier_count):
    clf = AdaBoostClassifier(classifier,n_estimators=classifier_count)
    clf.fit(xtrain,ytrain)
    yout = clf.predict(xtest)
    return yout

def readData():
    DATA_DIR = 'dir_data/'
    DATA_DIR = '/home/vinaykola/vinaykola/Acads/PatternRecognition/dir_data/'
    TEST_DIR = DATA_DIR + 'test/'
    TRAIN_DIR = DATA_DIR + 'train/'
    VALIDATION_DIR = DATA_DIR + 'validation/'
    data_types = ['train','test','validation']
        
    d_dir = []
    d_dir.append(TRAIN_DIR)
    d_dir.append(TEST_DIR)
    d_dir.append(VALIDATION_DIR)
        
    features = {}
    features['audio'] = ['audio_' + x for x in ['mfcc','sai_boxes','sai_intervalgrams','spectrogram_stream','volume_stream']]
    features['text'] = ['text_' + x for x in ['description_unigrams','game_lda_1000','tag_unigrams']]
    features['vision'] = ['vision_' + x for x in ['cuboids_histogram','hist_motion_estimate','hog_features','hs_hist_stream','misc']]

    all_features = features['audio'] + features['text'] + features['vision']    
    feature_families = ['audio','text','vision']
    file_names = get_file_names(d_dir,data_types,features,feature_families)
    print file_names
    print len(file_names['train'])
    all_filenames = file_names['train'] + file_names['test'] + file_names['validation'] 
    feature_sizes = load_variable('sizes.pkl',get_feature_range,file_names,all_features)
    print feature_sizes
    all_feature_sizes = [feature_sizes[feature] for feature in all_features]
    families = []
    for i in xrange(5):
        families.append('audio')
    for i in xrange(3):
        families.append('text')
    for i in xrange(5):
        families.append('vision')

    n = {}
    n['train'] = 97935
    n['test'] = 11931
    n['validation'] = 12177

#    data_sizes = [n[family] for family in families]#
    
    data_type = 'validation'
    test_arrays = read_files(data_type,file_names[data_type], all_features, families, n[data_type], all_feature_sizes)

def read_files(data_type, all_filenames, all_features, families, data_size, feature_sizes):    
    #create numpy arrays for each file
    arrays = []
    PICKLE_DIR = '/home/vinaykola/vinaykola/Acads/PatternRecognition/pickle_data/'
    i = 0
    for f_name,feature,y in zip(all_filenames,all_features,feature_sizes):
        print i
        #if i < 5:
        #    i += 1
        #    print 'skipping ..'
        #    continue
        pickle_filename = data_type + "_" + feature + '.pkl'
        print pickle_filename
        array = load_variable(PICKLE_DIR + pickle_filename,read_file,f_name,data_type,(data_size,y))
        arrays.append(array)
        #print 'whoa'
        i += 1
    return arrays

# Size = [#(data_points,#(no_of_features)]
# Returns an array of size [#(data_points,#(no_of_features) + 1]
def read_file(f_name,data_type,size):


    # Instantiate a lil_matrix because inserts are faster
    size = (int(size[0]),int(size[1])+1)
    if size[1] > 10000:
        array = scipy.sparse.lil_matrix(size)
    else:
        array = np.zeros(size)
    
    with open(f_name,'r') as f:
        for i,line in enumerate(f):
            if i % 100 == 0:
                print i
            if i % 2 == 0:
                try:
                    _,id = line.split('\t')
                    id = int(id)
                except ValueError:
                    break                
            else:
                fs = line.split(' ')
                video_class = fs[0]
                array[id-firstID(data_type),size[1]-1] = fs[0]
                for feature in fs[1:]:
                    k,v = feature.split(':')
                    # Feature numbers start from 1, so decrement by 1
                    array[id-firstID(data_type),int(k)-1] = float(v)

    if size[1] > 10000:
        # Return the CSR version of the matrix
        return array.tocsr()
    else:
        return array

                        
# Get file names for all train and test files
def get_file_names(d_dir,data_types,features,feature_families):
    file_names = {}
    count = 0
    for dir_type,data_type in zip(d_dir,data_types):
        file_names[data_type] = []
        for f_family in feature_families:
            for f_type in features[f_family]:
                file_names[data_type].append(dir_type + f_type + '.txt')
                count += 1
    return file_names

# Get number of features for all feature types 
def get_feature_range(file_names, features):

    sizes = {}
    
    for file_name,feature_name in zip(file_names['test'],features):
        with open(file_name,'r') as f:
            cols = 0
            for i,line in enumerate(f):
               if i % 2 != 0:
                    fs = line.split(' ')
                    for feature in fs[1:]:
                        k,v = feature.split(':')
                        if int(k) > cols:
                            cols = int(k) 
            print file_name
            print feature_name
            print cols
            sizes[feature_name] = cols

    return sizes
        


def firstID(data_type):
    if data_type == 'train':
        return 1
    elif data_type == 'test':
        return 300001
    elif data_type == 'validation':
        # Check this number in the data
        return 200001
    else:
        return None


def main():
    readData()
        
if __name__ == '__main__':
    main()

