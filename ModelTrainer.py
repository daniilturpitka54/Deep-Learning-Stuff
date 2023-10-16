import numpy
from keras.models import Model, load_model, save_model
from keras.callbacks import History
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
import os, sys
import random
import threading
from keras.layers import Input, Dense, Conv2D, UpSampling2D, Conv2DTranspose, Flatten, MaxPooling2D, BatchNormalization, SpatialDropout2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from sklearn.linear_model import LogisticRegression
n = 0
directory='Stamps'
color0 = None
colorrg = None
colorbg = None
train0= valid0 = None
trainrg=validrg = None
trainbg= validbg = None
filenames = []
aerb = None
aerg = None
aebg = None
inprb=encodedrb=inprg=encodedrg=inpbg=encodedbg = None

def readFileNames():
    global filenames 
    c = 0
    dirlist = os.listdir(directory)
    global colorrb, colorrg, colorbg
    for i in  dirlist:
        try:
            for j in os.listdir(os.path.join(directory,i)):
                filename = os.path.join(directory, i, j)
                filenames.append(filename)
                print("filenames read:  ",len(filenames), end='\r' )
        except Exception as e:
            print(e)
            pass
        if c>5000:
            break
    return None


def read_validation_files(n):
    global  valid0,  validrg,  validbg
    global colorrb, colorrg, colorbg
    colorrb = numpy.zeros((n,256,256))
    colorrg = numpy.zeros((n,256,256))
    colorbg = numpy.zeros((n,256,256))
    valid0 = numpy.zeros((n,256,256))
    validrg = numpy.zeros((n,256,256))
    validbg = numpy.zeros((n,256,256))
    dirs = [random.randint(1,len(filenames)) for _ in range(n)]
    c = 0
    for i in dirs:
        img = load_img(filenames[i])
        img = numpy.array(img)
        valid0[c]=img[:,:,0]/255
        #f, (ax1, ax2, ax3) = plt.subplots(3,1)
        #ax1.imshow(validrb[c])
        #validrg[c]=numpy.subtract(img[:,:,0],img[:,:,2])/255
        #ax2.imshow(validrg[c])
        #validbg[c]=numpy.subtract(img[:,:,1],img[:,:,2])/255
        #ax3.imshow(validbg[c])
        del img
        print("files read:  ", str(c)+'\r',end='')
        c+=1
    #trainrb, validrb = train_test_split(colorrb, random_state=32, test_size=0.15)
    #trainrg, validrg = train_test_split(colorrg, random_state=32, test_size=0.15)
    #trainbg, validbg = train_test_split(colorbg, random_state=32, test_size=0.15)
    print('\n')
    #print(validrb[0])


def read_train_files(n):
    global train0, trainrg, trainbg
    #global color0, color1, color2
    train0 = numpy.zeros((n,256,256))
    trainrg = numpy.zeros((n,256,256))
    trainbg = numpy.zeros((n,256,256))
    dirs = [random.randint(1,len(filenames)) for _ in range(n)]
    c = 0
    for i in dirs:
        img = load_img(filenames[i])
        img = numpy.array(img)
        train0[c]=img[:,:,0]/255
        #trainrg[c]=numpy.subtract(img[:,:,0], img[:,:,2])/255
        #trainbg[c]=numpy.subtract(img[:,:,1], img[:,:,2])/255
        del img
        print("files read:  ", str(c)+'\r',end='')
        c+=1
    # train0 = color0
    # train1 = color1
    # train2 = color2
    print('\n')
    # train0, valid0 = train_test_split(color0, random_state=32, test_size=0.15)
    # train1, valid1 = train_test_split(color1, random_state=32, test_size=0.15)
    # train2, valid2 = train_test_split(color2, random_state=32, test_size=0.15)


def compileColor0(epochs=400, bs=64 ):
    global aerb, inprb, encodedrb
    inprb = Input(shape=(256,256,1))

    conv01 = Conv2D(filters=12, kernel_size=(4,4), activation='relu', padding='same')(inprb)
    norm1=BatchNormalization(epsilon=0.01)(conv01)
    #norm1=SpatialDropout2D(0.1)(norm1)
    maxpol01 = MaxPooling2D(pool_size=(2,2), padding='same')(norm1)
    
    conv02 = Conv2D(filters=8, kernel_size=(4,4), activation='relu', padding='same')(maxpol01)
    norm2=BatchNormalization(epsilon=0.01)(conv02)
    #norm2=SpatialDropout2D(0.1)(norm2)
    maxpol02 = MaxPooling2D(pool_size=(2,2), padding='same')(norm2)

    conv03 = Conv2D(filters=6, kernel_size=(3,3), activation='relu', padding='same')(maxpol02)
    norm3=BatchNormalization(epsilon=0.01)(conv03)
    #norm3=SpatialDropout2D(0.1)(norm3)
    maxpol03 = MaxPooling2D(pool_size=(2,2), padding='same')(norm3)

    conv04 = Conv2D(filters = 4, kernel_size = (2,2), activation = 'relu', padding='same')(maxpol03)
    maxpol04 = MaxPooling2D(pool_size = (2,2), padding='same')(conv04)

    conv05 = Conv2D(filters = 2, kernel_size = (2,2), activation = 'relu', padding='same')(maxpol04)
    encodedrb = MaxPooling2D(pool_size = (2,2), padding='same')(conv05)


    deconv0 = Conv2DTranspose(filters = 2, kernel_size=(2,2), activation='relu', padding='same')(encodedrb)
    upsamp00 = UpSampling2D(size=(2,2))(deconv0)

    deconv01 = Conv2DTranspose(filters = 4, kernel_size=(2,2), activation='relu', padding='same')(upsamp00)
    upsamp01 = UpSampling2D(size=(2,2))(deconv01)

    deconv02 = Conv2DTranspose(filters = 6, kernel_size=(3,3), activation='relu', padding='same')(upsamp01)
    upsamp02 = UpSampling2D(size=(2,2))(deconv02)

    deconv03 = Conv2DTranspose(filters = 8, kernel_size=(4,4), activation='relu', padding='same')(upsamp02)
    upsamp03 = UpSampling2D(size=(2,2))(deconv03)

    deconv04 = Conv2DTranspose(filters = 12, kernel_size=(4,4), activation='relu', padding='same')(upsamp03)
    upsamp04 = UpSampling2D(size=(2,2))(deconv04)

    decoded0 = Conv2DTranspose(filters=1, kernel_size=(4,4), activation='sigmoid', padding='same')(upsamp04)

    aerb = Model(inprb, decoded0)
    aerb.compile(optimizer = "RMSprop", loss='mse', metrics='accuracy')
    aerb.summary()
    #ae0.fit(train0, train0, epochs=epochs, batch_size=64, shuffle=True, validation_data =(valid0,valid0), verbose=1 )
    # ae0.save("model_256x256_c0")
    # encoder = Model(inp0, encoded0)
    # encoder.save('model_256x256_c0_encoder')
    return 

def trainColor0(epochs=350, bs=64):
    global trainrb, validrb, aerb, inprb, encodedrb
    tx = 'ColorRB'
    print(f'{tx:#>10}')
    aerb.fit(train0, train0, epochs=epochs, batch_size=bs, shuffle=True, validation_data =(valid0,valid0), verbose=1 )
    #ae0.save("model_256x256_c0")
    #encoder = Model(inp0, encoded0)
    #encoder.save('model_256x256_c0_encoder')



def compileColor1(epochs=400, bs=64 ):
    global aerg, inprg, encodedrg
    inprg = Input(shape=(256,256,1))

    conv1 = Conv2D(filters=8, kernel_size=(4,4), activation='relu', padding='same')(inprg)
    x = BatchNormalization(epsilon=0.001)(conv1)
    x = SpatialDropout2D(0.2)(x)
    maxpol1 = MaxPooling2D(pool_size=(2,2), padding='same')(x)

    conv2 = Conv2D(filters=6, kernel_size=(4,4), activation='relu', padding='same')(maxpol1)
    x = BatchNormalization(epsilon=0.001)(conv2)
    x = SpatialDropout2D(0.1)(x)
    maxpol2 = MaxPooling2D(pool_size=(2,2), padding='same')(x)

    conv3 = Conv2D(filters=4, kernel_size=(3,3), activation='relu', padding='same')(maxpol2)
    x = BatchNormalization(epsilon=0.001)(conv3)
    x = SpatialDropout2D(0.1)(x)
    maxpol3 = MaxPooling2D(pool_size=(2,2), padding='same')(x)

    conv4 = Conv2D(filters = 4, kernel_size = (6,6), activation = 'relu', padding='same')(maxpol3)
    maxpol4 = MaxPooling2D(pool_size = (2,2), padding='same')(conv4)

    conv5 = Conv2D(filters = 2, kernel_size = (6,6), activation = 'relu', padding='same')(maxpol4)
    encodedrg = MaxPooling2D(pool_size = (2,2), padding='same')(conv5)


    deconv0 = Conv2DTranspose(filters = 2, kernel_size=(6,6), activation='relu', padding='same')(encodedrg)
    upsamp0 = UpSampling2D(size=(2,2))(deconv0)

    deconv1 = Conv2DTranspose(filters = 4, kernel_size=(6,6), activation='relu', padding='same')(upsamp0)
    upsamp1 = UpSampling2D(size=(2,2))(deconv1)

    deconv2 = Conv2DTranspose(filters = 4, kernel_size=(3,3), activation='relu', padding='same')(upsamp1)
    upsamp2 = UpSampling2D(size=(2,2))(deconv2)

    deconv3 = Conv2DTranspose(filters = 6, kernel_size=(4,4), activation='relu', padding='same')(upsamp2)
    upsamp3 = UpSampling2D(size=(2,2))(deconv3)

    deconv4 = Conv2DTranspose(filters = 6, kernel_size=(4,4), activation='relu', padding='same')(upsamp3)
    upsamp4 = UpSampling2D(size=(2,2))(deconv4)

    decoded = Conv2DTranspose(filters=1, kernel_size=(4,4), activation='sigmoid', padding='same')(upsamp4)

    aerg = Model(inprg, decoded)
    aerg.compile(optimizer = "RMSprop", loss='mse', metrics='accuracy')
    # ae1.fit(train1, train1, epochs=epochs, batch_size=64, shuffle=True, validation_data =(valid0,valid0), verbose=1 )
    
    # ae1.save("model_256x256_c1")
    # encoder = Model(inp1, encoded1)
    # encoder.save('model_256x256_c1_encoder')
    return 


def trainColor1(epochs=400, bs=64):
    global train1, valid1, aerg, inprg, encodedrg
    tx = 'ColorRG'
    print(f'{tx:#>10}')
    aerg.fit(trainrg, trainrg, epochs=epochs, batch_size=bs, shuffle=True, validation_data =(validrg,validrg), verbose=1 )
    #ae0.save("model_256x256_c0")
    #encoder = Model(inp1, encoded1)
    #encoder.save('model_256x256_c1_encoder')
    return


def compileColor2():
    global trainbg, validbg, inpbg, encodedbg, aebg
    inpbg = Input(shape=(256,256,1))

    conv1 = Conv2D(filters=8, kernel_size=(4,4), activation='relu', padding='same')(inpbg)
    x = BatchNormalization(epsilon=0.001)(conv1)
    x = SpatialDropout2D(0.2)(x)
    maxpol1 = MaxPooling2D(pool_size=(2,2), padding='same')(x)

    conv2 = Conv2D(filters=6, kernel_size=(4,4), activation='relu', padding='same')(maxpol1)
    x = BatchNormalization(epsilon=0.001)(conv2)
    x = SpatialDropout2D(0.1)(x)
    maxpol2 = MaxPooling2D(pool_size=(2,2), padding='same')(x)

    conv3 = Conv2D(filters=4, kernel_size=(3,3), activation='relu', padding='same')(maxpol2)
    x = BatchNormalization(epsilon=0.001)(conv3)
    x = SpatialDropout2D(0.1)(x)
    maxpol3 = MaxPooling2D(pool_size=(2,2), padding='same')(x)

    conv4 = Conv2D(filters = 4, kernel_size = (6,6), activation = 'relu', padding='same')(maxpol3)
    maxpol4 = MaxPooling2D(pool_size = (2,2), padding='same')(conv4)

    conv5 = Conv2D(filters = 2, kernel_size = (6,6), activation = 'relu', padding='same')(maxpol4)
    encodedbg = MaxPooling2D(pool_size = (2,2), padding='same')(conv5)


    deconv0 = Conv2DTranspose(filters = 2, kernel_size=(6,6), activation='relu', padding='same')(encodedbg)
    upsamp0 = UpSampling2D(size=(2,2))(deconv0)

    deconv1 = Conv2DTranspose(filters = 4, kernel_size=(6,6), activation='relu', padding='same')(upsamp0)
    upsamp1 = UpSampling2D(size=(2,2))(deconv1)

    deconv2 = Conv2DTranspose(filters = 4, kernel_size=(3,3), activation='relu', padding='same')(upsamp1)
    upsamp2 = UpSampling2D(size=(2,2))(deconv2)

    deconv3 = Conv2DTranspose(filters = 6, kernel_size=(4,4), activation='relu', padding='same')(upsamp2)
    upsamp3 = UpSampling2D(size=(2,2))(deconv3)

    deconv4 = Conv2DTranspose(filters = 6, kernel_size=(4,4), activation='relu', padding='same')(upsamp3)
    upsamp4 = UpSampling2D(size=(2,2))(deconv4)

    decoded = Conv2DTranspose(filters=1, kernel_size=(4,4), activation='sigmoid', padding='same')(upsamp4)

    aebg = Model(inpbg, decoded)
    aebg.compile(optimizer = "RMSprop", loss='mse', metrics='accuracy')
    #ae1.fit(train2, train2, epochs=epochs, batch_size=64, shuffle=True, validation_data =(valid2,valid2), verbose=1 )
    
    #ae1.save("model_256x256_c2")
    #encoder = Model(inp2, encoded2)
    #encoder.save('model_256x256_c2_encoder')
    return 


def trainColor2(epochs=400, bs=64):
    global trainbg, valid2, aebg, inpbg, encodedbg
    tx = 'ColorBG'
    print(f'{tx:#>10}')
    aebg.fit(trainbg, trainbg, epochs=epochs, batch_size=bs, shuffle=True, validation_data =(validbg,validbg), verbose=1 )
    #ae0.save("model_256x256_c2")
    #encoder = Model(inp1, encoded1)
    #encoder.save('model_256x256_c2_encoder')
    return


def saveModels():
    global aerb, inprb, encodedrb, aerg, inprg, encodedrg, aebg, inpbg, encodedbg
    aerb.save("model_256x256_c0")
    encoder = Model(inprb, encodedrb)
    encoder.save('model_256x256_c0_encoder')

    # aerg.save("model_256x256_crg")
    # encoder = Model(inprg, encodedrg)
    # encoder.save('model_256x256_crg_encoder')

    # aebg.save("model_256x256_cbg")
    # encoder = Model(inpbg, encodedbg)
    # encoder.save('model_256x256_cbg_encoder')

def main(n=45000, batches=3):
    global trainbg, trainrb, trainrg
    readFileNames()
    start = 0
    compileColor0()
    compileColor1()
    compileColor2()
    nval = int(n*0.15)
    read_validation_files(nval)
    bs = int(n/batches)
    batchsize=64
    for _ in range(batches):
        read_train_files(n)
        trainColor0(15, batchsize)
        #trainColor1(15,batchsize)
        #trainColor2(15,batchsize)
        #batchsize =batchsize*2

    
    saveModels()

if __name__=='__main__':
    main(20000, 1)
    
