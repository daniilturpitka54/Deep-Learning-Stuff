import numpy, random
from utils import get_features, produce_features, produce_stds
from keras.models import Model, load_model, save_model
from keras.callbacks import History
import matplotlib.pyplot as plt
from keras.preprocessing import image
import os
from keras.layers import Input, Dense, Conv2D, UpSampling2D, Conv2DTranspose, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

ae0 = load_model('model_256x256_c0')
encoder0 = load_model('model_256x256_c0_encoder')


ae1 = load_model('model_256x256_c1')
encoder1 = load_model('model_256x256_c1_encoder')

ae2 = load_model('model_256x256_c1')
encoder2 = load_model('model_256x256_c1_encoder')

aerb = load_model('model_256x256_crb')
encoderrb = load_model('model_256x256_crb_encoder')

aerg = load_model('model_256x256_crg')
encoderrg = load_model('model_256x256_crg_encoder')

aebg = load_model('model_256x256_cbg')
encoderbg = load_model('model_256x256_cbg_encoder')


directory = 'FirstImageSet/2021-GraL-ImageSet-gri/Lens'
lenses = []
dirlist = os.listdir(directory)
original_images = numpy.zeros((len(dirlist), 256,256,3))
#validlens = original_images[180:]
#original_images = original_images[:180]


qso0 = qso1 = qso2 = qsorb = qsorg = qsobg = numpy.zeros((800,256,256))
lensc0 = lensc1 = lensc2 = lenscrb = lenscrg = lenscbg = numpy.zeros((len(dirlist), 256,256))
for i in range(len(dirlist)):
    filename = os.path.join(directory, dirlist[i])
    img = numpy.array(image.load_img(filename))
    lensc0[i] = img[:,:,0]/255
    lensc1[i] = img[:,:,1]/255
    lensc2[i] = img[:,:,2]/255
    lenscrb[i] = numpy.subtract(img[:,:,0],img[:,:,1])/255
    lenscrg[i] = numpy.subtract(img[:,:,0],img[:,:,2])/255
    lenscbg[i] = numpy.subtract(img[:,:,1],img[:,:,2])/255
lenses = numpy.array(lenses)
#print(lenses.shape)

directoryqso = 'FirstImageSet/2021-GraL-ImageSet-gri/QSO'
dirlist = os.listdir(directoryqso)
qsos = numpy.zeros((800, 256,256,3))

indxs = [random.randint(0, len(dirlist)) for _ in range(800)]
c = 0
for i in indxs:
    filename = os.path.join(directoryqso, dirlist[i])
    img = numpy.array(image.load_img(filename))
    qso0[c] = img[:,:,0]/255
    qso1[c] = img[:,:,1]/255
    qso2[c] = img[:,:,2]/255
    qsorb[c] = numpy.subtract(img[:,:,0],img[:,:,1])/255
    qsorg[c] = numpy.subtract(img[:,:,0],img[:,:,2])/255
    qsobg[c] = numpy.subtract(img[:,:,1],img[:,:,2])/255
    c+=1


qsolabels = numpy.zeros((800))
lenslabels = numpy.ones((226))
qsof0 = get_features(qso0, ae0,encoder0)
qsof1 = get_features(qso1, ae1, encoder1)
qsof2 = get_features(qso2, ae2, encoder2)
qsofrb = get_features(qsorb, aerb, encoderrb)
qsofrg = get_features(qsorg,aerg, encoderrg)
qsofbg = get_features(qsobg,aebg, encoderbg)

lensf0 = get_features(lensc0, ae0, encoder0)
lensf1 = get_features(lensc1, ae1, encoder1)
lensf2 = get_features(lensc2, ae2, encoder2)
lensfrb = get_features(lenscrb, aerb, encoderrb)
lensfrg = get_features(lenscrg, aerg, encoderrg)
lensfbg = get_features(lenscbg, aebg, encoderbg)


tfeatures = []
for i in range(600):
    temp = numpy.concatenate((qsof0[i], qsof1[i], qsof2[i], qsofrb[i], qsofrg[i], qsofbg[i]))
    #, qsofrb[i], qsofrg[i], qsofbg[i]
    #print(temp.shape)
    tfeatures.append(temp)
for i in range(200):
    temp = numpy.concatenate((lensf0[i], lensf1[i], lensf2[i], lensfrb[i], lensfrg[i], lensfbg[i]))
    #, lensfrb[i], lensfrg[i], lensfbg[i]
    tfeatures.append(temp)
tlabels = [0]*600
tmp = [1]*200
tlabels+=tmp
vfeatures = []
for i in range(600,800):
    temp = numpy.concatenate((qsof0[i], qsof1[i], qsof2[i]))
    vfeatures.append(temp)
for i in range(200,225):
    temp = numpy.concatenate((lensf0[i], lensf1[i], lensf2[i]))
    vfeatures.append(temp)
vlabels = [0]*200
tmp = [1]*25
vlabels+=tmp
    

tfeatures = numpy.array(tfeatures)
xgbc = XGBClassifier(alpha=1, learning_rate = 0.3)
#print(tfeatures.shape)
#print(tlabels)
xgbc.fit(tfeatures, tlabels)
xgbc.save_model('XGB6.json')

a = xgbc.predict(vfeatures)
print('confusion matrix for  3 collors and 3 collor differences')
print(confusion_matrix(vlabels, a))
print(accuracy_score(vlabels, a))

#reshape
