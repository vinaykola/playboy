from __future__ import division
import numpy as np

DATA_DIR = '/home/vinaykola/vinaykola/Acads/PatternRecognition/dir_data/'
TEST_DIR = DATA_DIR + 'test/'
TRAIN_DIR = DATA_DIR + 'train/'
VALIDATION_DIR = DATA_DIR + 'validation/'

features = {}
features['audio'] = ['audio_' + x for x in ['mfcc','sai_boxes','sai_intervalgrams','spectrogram_stream','volume_stream']]
features['text'] = ['text_' + x for x in ['description_unigrams','game_lda_1000','tag_unigrams']]
features['vision'] = ['vision_' + x for x in ['cuboids_histogram','hist_motion_estimate','hog_features','hs_hist_stream','misc']]

file_name = TEST_DIR + features['audio'][0] + '.txt'

test_size = 11931
audio_mfcc_features = 2000

audio_mfcc_test = np.zeros((test_size,audio_mfcc_features))

with open(file_name,'r') as f:
    for i,line in enumerate(f):
        #raw_input()
        if i % 2 == 0:
            try:
                _,id = line.split('\t')
                id = int(id)
                print id
            except ValueError:
                break
                
        else:
            fs = line.split(' ')
            video_class = fs[0]
            for feature in fs[1:]:
                k,v = feature.split(':')
                k = int(k) - 1
                v = float(v)
                audio_mfcc_test[id-300001,k] = v
