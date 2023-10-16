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

ae, encoder = None, None
xgbclassifier = XGBClassifier()
preds = []
predicted_lenses = open('predicted_lenses.txt', 'w')

def loadModels():
    global ae, encoder
    ae = load_model('model_sp23')
    encoder = load_model('model_256x256_sp23_enc')
    xgbclassifier.load_model('xgbclassifier.json')
    return None


def produce_stds(original_images, autoencoder):
    color_prediction = autoencoder.predict(original_images)
    original_images = original_images.reshape((len(original_images),256,256,3))
    color_prediction = color_prediction.reshape((len(original_images),256,256,3))
    residuals = (original_images - color_prediction)**2
    residuals = numpy.array(residuals)
    #print(residuals.shape)
    return numpy.array([numpy.std(i) for i in residuals])

def produce_features(original_images, encoder, residuals):
    compressed = numpy.array(encoder.predict(original_images))
    compressed = compressed.reshape(len(original_images), 128)
    return numpy.array([numpy.append(compressed[i], residuals[i]) for i in range(len(original_images))])

def get_features(original_images, color):
    stds = produce_stds(original_images, autoencoders[color])
    return produce_features(original_images, encoders[color], stds)

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
    c = os.listdir(directory)
    print(c[:60])
    for i in os.listdir(directory):
        try:
            for j in os.listdir(os.path.join(directory,i)):
                
                filename = os.path.join(directory, i, j)
                #print(os.path.join(directory,i))
                img = image.load_img(filename)
                img = numpy.array(img)
                color0=[]
                color0.append(img[:,:,0]/255)
                color0 = numpy.array(color0)
                color1=[]
                color1.append(img[:,:,1]/255)
                color1 = numpy.array(color1)
                color2=[]
                color2.append(img[:,:,2]/255)
                color2 = numpy.array(color2)
                
                #creating features from an immage
                #print(color0.shape)
                color0_res = produce_stds(color0, ae0)
                
                color0_features = produce_features(color0, encoder0, color0_res)
                color1_res = produce_stds(color1, ae1)
                color1_features = produce_features(color1, encoder1, color1_res)
                color2_res = produce_stds(color2, ae2)
                color2_features = produce_features(color2, encoder2, color2_res)
                features = []
                temp = color0_features[0]
                temp += color1_features[0]
                temp +=color2_features[0]
                features.append(temp)
                pred = xgbclassifier.predict_proba(features)
                if pred[0][1]>prob:
                    #predicted_lenses.write(filename+'\n')
                    preds.append((filename, pred[0][1]))
                    print(c)
                    print(f"Lense found! prob={pred[0][1]}")
                else:
                    print(c, end='\r')
                c+=1
                if c>numfiles:
                    #predicted_lenses.close()
                    return

        except Exception as e:
            print(e)
            pass
    predicted_lenses.close()


def write_file():
    global preds, predicted_lenses

    preds = sorted(preds, key=lambda x: -x[1])
    for i in preds:
        predicted_lenses.write(str(i)+'\n')
    predicted_lenses.close()


def main(prob, numfiles):
    loadModels()
    readfile(prob,numfiles )
    write_file()

if __name__=='__main__':
    main(float(sys.argv[1]), int(sys.argv[2]))