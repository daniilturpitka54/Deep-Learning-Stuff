import numpy
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import os, sys
from keras.layers import Input, Dense, Conv2D, UpSampling2D, Conv2DTranspose, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model, load_model
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from utils import get_features
ae0 = None
ae1 = None
ae2 = None
predicted_lenses = None
aerb = aerg = aebg = None
encoderrb = encoderrg = encoderbg = None
encoder0 = None
encoder1 = None
encoder2 = None
autoencoders = None
encoders = None
xgbclassifier = XGBClassifier()
preds = []


def loadModels():
    global ae0
    global ae1
    global ae2
    global aerb,aerg,aebg,encoderrb,encoderrg,encoderbg
    global encoder0
    global encoder1
    global encoder2
    global autoencoders
    global e
    global encoders
    global xgbclassifier
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

    aerg = load_model('model_256x256_crg')
    encoderrg = load_model('model_256x256_crg_encoder')

    aebg = load_model('model_256x256_cbg')
    encoderbg = load_model('model_256x256_cbg_encoder')
    #autoencoders = {0:ae0, 1:ae1, 2:ae2}
    #encoders = {0:encoder0, 1:encoder1, 2:encoder2}
    xgbclassifier.load_model('XGB6.json')
    return None



def reconstruct_images(c0, c1, c2):
    ret = []
    for i in range(len(c0)):
        ar = numpy.empty((256,256,3))
        #c0[i] = c0[i].reshape(64,64)
        #c1[i] = c0[i].reshape(64,64)
        #c2[i] = c2[i].reshape(64,64)
        ar[:,:,0] = c0[i].reshape(256,256)
        ar[:,:,1] = c1[i].reshape(256,256)
        ar[:,:,2] = c2[i].reshape(256,256)
        ret.append(ar)
    ret = numpy.array(ret)
    return ret


def readfile(prob, numfiles):
    global predicted_lenses, preds
    directory='Stamps'
    c = 0
    ce = 0
    
    dirlist = os.listdir(directory)
    #lensc0 = lensc1 = lensc2 = lenscrb = lenscrg = lenscbg = numpy.zeros((len(dirlist), 256,256))
    try:
        for i in dirlist :
            try:
                for j in os.listdir(os.path.join(directory,i)):
                    filename = os.path.join(directory, i, j)
                    #print(filename)
                    img = numpy.array(image.load_img(filename))
                    color0=[]
                    color0.append(img[:,:,0]/255)
                    color0 = numpy.array(color0)
                    color1=[]
                    color1.append(img[:,:,1]/255)
                    color1 = numpy.array(color1)
                    color2=[]
                    color2.append(img[:,:,2]/255)
                    color2 = numpy.array(color2)
                    colorrb = [numpy.zeros((1,256,256))]
                    colorrb[0] = numpy.subtract(img[:,:,0],img[:,:,1])/255
                    colorrb = numpy.array(colorrb)
                    colorrg = [numpy.zeros((1,256,256))]
                    colorrg[0] = numpy.subtract(img[:,:,0],img[:,:,2])/255
                    colorrg = numpy.array(colorrg)
                    colorbg = [numpy.zeros((1,256,256))]
                    colorbg[0]=numpy.subtract(img[:,:,1],img[:,:,2])/255
                    colorbg = numpy.array(colorbg)
                    #creating features from an immage
                    #print(color0.shape)
                    
                    features = []
                    f0 = get_features(color0, ae0, encoder0)[0]
                    f1  = get_features(color1, ae1, encoder1)[0]
                    f2 =get_features(color2, ae2, encoder2)[0]
                    frb=get_features(colorrb, aerb, encoderrb)[0]
                    frg=get_features(colorrg, aerg, encoderrg)[0]
                    fbg=get_features(colorbg, aebg, encoderbg)[0]
                    temp = numpy.concatenate((f0,f1,f2,frb,frg,fbg))
                    features.append(temp)
                    pred = xgbclassifier.predict_proba(features)
                    #print(pred)
                    if pred[0][1]>prob:
                        #predicted_lenses.write(filename+'\n')
                        #preds.append((filename, pred[0][1]))
                        predicted_lenses.write(f"{j},{str(pred[0][1])}"+'\n')
                        #print(c)
                        print(f"Lense found! prob={pred[0][1]}, {c}")
                    print(f"files read: {c}, curdir: {i}",  end='\r')
                    c+=1
                    del color0, color1, color2, colorrb, colorbg, colorrg, img, temp, features


            except Exception as e:
                ce+=1
                print(f"Error reading directory: {e}, errornum: {ce}")
                
                pass
    except Exception as e:
        predicted_lenses.close()
        pass
    predicted_lenses.close()



def write_file(numlens):
    global preds
    # predicted_lenses = open('predicted_lenses2.txt', 'w')
    preds = sorted(preds, key=lambda x: -x[1])
    for i in range(numlens):
        predicted_lenses.write(str(preds[i])+'\n')
    predicted_lenses.close()
    #predicted_lenses.close()


def main(prob, numfiles):
    global predicted_lenses
    try:
        predicted_lenses = open('predicted_lensesxgb63.txt', 'w')
        loadModels()
        readfile(prob,numfiles )
    except Exception as exc:
        predicted_lenses.close()
        print(exc)
    #write_file(numfiles)

if __name__=='__main__':
    main(0.85, 100)