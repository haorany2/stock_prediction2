
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import Model
from keras import optimizers
from keras.layers import Concatenate,Flatten,Add,Activation,BatchNormalization,Conv2D,Dense, Dropout, LSTM,Input,ZeroPadding2D,AveragePooling2D,MaxPooling2D
from keras.initializers import glorot_uniform
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.losses import mse
import os
import h5py
from mpl_finance import candlestick_ohlc
from mpl_finance import candlestick2_ohlc
from mpl_finance import volume_overlay2
from mpl_finance import volume_overlay
from mpl_finance import index_bar
from sklearn.preprocessing import MinMaxScaler

import cv2
import pickle

epochs = 16
batch_size_is = 16

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    
    
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform())(X_shortcut)
    
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X_shortcut,X])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X



def combine_model(multi_gpu=True,num_gpus=4):
    img_X_input=Input(shape=(112,112,3))
    lstm_X_input = Input(shape=(29,2))
    img_X = Conv2D(32, (7, 7), padding='same',strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform())(img_X_input)
    #img_X = BatchNormalization(axis = 3, name = 'bn_conv1')(img_X)
    img_X = Activation('relu')(img_X)
    img_X = MaxPooling2D((3, 3),strides=(2, 2),padding='same')(img_X)
    img_X = convolutional_block(img_X, f = 3, filters = [32, 32, 128], stage = 2, block='a', s = 1)
    img_X = convolutional_block(img_X, f = 3, filters = [64, 64, 256], stage = 3, block='a', s = 2)
    img_X = convolutional_block(img_X, f = 3, filters = [128, 128, 512], stage = 4, block='a', s = 2)
    img_X=AveragePooling2D(pool_size=(7, 7), strides=None, padding='valid')(img_X)
    img_X=Flatten()(img_X)
    img_X=Dense(500,activation='relu')(img_X)
    
    #img_X_more=img_X
    img_X_more=Dropout(0.5)(img_X)
    img_X_more=Dense(100,activation='relu')(img_X_more)
    img_X_more=Dropout(0.5)(img_X_more)
    img_X_more=Dense(25,activation='relu')(img_X_more)
    img_X_more=Dropout(0.5)(img_X_more)
    img_X_more=Dense(1)(img_X_more)
    
    
    lstm_X=LSTM(units=500, return_sequences=True)(lstm_X_input)
    lstm_X=LSTM(units=500)(lstm_X)
    lstm_X=Dense(500,activation='relu')(lstm_X)
    
    #lstm_X_more=lstm_X
    lstm_X_more=Dropout(0.5)(lstm_X)
    lstm_X_more=Dense(100,activation='relu')(lstm_X_more)
    lstm_X_more=Dropout(0.5)(lstm_X_more)
    lstm_X_more=Dense(25,activation='relu')(lstm_X_more)
    lstm_X_more=Dropout(0.5)(lstm_X_more)
    lstm_X_more=Dense(1)(lstm_X_more)
    #consider output will reduce neuron again 
    fused=Concatenate()([img_X,lstm_X])
    fused=Dense(500,activation='relu')(fused)
    fused=Dropout(0.5)(fused)
    fused=Dense(100,activation='relu')(fused)
    fused=Dropout(0.5)(fused)
    fused=Dense(25,activation='relu')(fused)
    fused=Dropout(0.5)(fused)
    fused=Dense(1)(fused)
    model = Model(inputs=[img_X_input,lstm_X_input],outputs=fused)
    
    def root_mean_squared_error(y_true, y_pred):
        #print ("y_ture: ", y_true)
        #print ("y_pred: ", y_pred)
        #print("img_X",img_X.shape)
        #img_X,lstm_X,fused=combine_model()
        #loss1=K.sqrt(K.mean(K.square(img_X_more - y_true), axis=-1)) 
        #loss2=K.sqrt(K.mean(K.square(lstm_X_more - y_true), axis=-1)) 
        #loss3=K.sqrt(K.mean(K.square(fused- y_true), axis=-1)) 
        loss1=mse(y_true,img_X_more)
        loss2=mse(y_true,lstm_X_more)
        loss3=mse(y_true,fused)
        #0.2*loss1+0.2*loss2+
        return 0.2*loss1+0.2*loss2+loss3
    
    
    
    
    print(model.summary())
    adam = optimizers.Adam(lr=1e-3, decay=5e-4)
    model.compile( loss=root_mean_squared_error, optimizer=adam)
    #return img_X,lstm_X,fused
    if(multi_gpu == True):
        parallel_model = multi_gpu_model(model, gpus=num_gpus)


    
        parallel_model.compile(optimizer= adam,
                      loss=root_mean_squared_error, 
                      metrics=root_mean_squared_error)
        return model, parallel_model
    
    
    return model



def lstm_gen(x_train_feat, target,train_batch_size):
    while True:
        for batch in range(x_train_feat.shape[0] // train_batch_size + 1):
            if batch > max(range(x_train_feat.shape[0] // train_batch_size)):
                yield x_train_feat[batch*train_batch_size:],target[batch*train_batch_size:]
            else:
                yield x_train_feat[batch*train_batch_size:(1+batch)*train_batch_size],target[batch*train_batch_size:(1+batch)*train_batch_size]


def img_gen(image_paths,  batch_size):  
    """
    
    Data generator for training data
    
    """      
         
    while True:
       
        for batch in range(len(image_paths) // batch_size + 1):
            X = []
           
            # choose random index in features
            #index = np.random.choice(len(image_paths),1)[0]
            if batch > max(range(len(image_paths) // batch_size)):
                for index in range (batch*batch_size,len(image_paths)):
            ##load image 
                    image = cv2.imread(image_paths[index])
                    image = cv2.resize(image,(112,112))
                    image=image*(1./225)
                    X.append(image)
            
                    #y.append(labels[image_paths[index]]) #add the label for the image   
    
                yield np.array(X)
            else:
                for index in range (batch*batch_size,(batch+1)*batch_size):
                    image = cv2.imread(image_paths[index])
                    image = cv2.resize(image,(112,112))
                    image=image*(1./225)
                    X.append(image)
                yield np.array(X)

def merge_generator(img_generator, lstm_generator):
        while True:
            X1 = img_generator.__next__()
            X2 = lstm_generator.__next__()
            yield [X1, X2[0]], X2[1]
            

            
df=pd.read_csv('tw_spydata_raw.csv')
#print the head
#df.head()

img_dir=os.getcwd()+'/candle_img'




lstm_data=df[['Trade Close', 'Trade Volume']].copy()
print("lstm_data: ",lstm_data.shape)

feature_lst=[]
target_lst=[]
img_dir_lst=[]
lstm_data=lstm_data.values.astype(float)

lstm_data_reducer=lstm_data[1:]
lstm_data_reduce=lstm_data[:-1]#latter, use lstm_data_reducer/lstm_data_reduce
log_return_fea=np.log(lstm_data_reducer)-np.log(lstm_data_reduce)

#scale with minmax
#scaler=MinMaxScaler((-1,1))
#scaler.fit(log_return_fea)
#log_return_fea=scaler.transform(log_return_fea)


lstm_data_reducer=lstm_data[34:,0]
lstm_data_reduce=lstm_data[29:-5,0]
log_return_tar=np.log(lstm_data_reducer)-np.log(lstm_data_reduce)
print("log_return_fea.shape[0]",log_return_fea.shape)
print("log_return_tar:",log_return_tar.shape)

#for i in range(0,len(data)):
#    new_data['Date'][i] = data['Date'][i]
#    new_data['Close'][i] = data['Close'][i]

#log_return_tar=log_return_fea[29:]
for i in range(29,log_return_fea.shape[0]-4):
    feature_lst.append(log_return_fea[i-29:i])
    target_lst.append(log_return_tar[i-29])
    img_dir_lst.append(img_dir+'/candle'+str(i-29)+'.png')
   

feature_lst=np.asarray(feature_lst)
target_lst=np.asarray(target_lst)


print("check nan:",np.isnan(target_lst).any())
print("check inf:",np.isinf(target_lst).any() )
#print(np.argwhere(np.isnan(target_lst)))
#img_dir_lst=np.asarray(img_dir_lst)
print("feature_lst shape (should same with target_lst): ",feature_lst.shape)
#np.save('lstm_input',feature_lst)
#np.save('target',target_lst)
#np.save('lstm_input',feature_lst)

train_size=int(feature_lst.shape[0]*0.7)
val_size=int(feature_lst.shape[0]*0.1)
test_size=feature_lst.shape[0]-train_size-val_size


print("---start to create pyh5 file---")
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('lstm_input_train', data=feature_lst[0:train_size])
h5f.create_dataset('lstm_input_val', data=feature_lst[train_size:train_size+val_size])
h5f.create_dataset('lstm_input_test', data=feature_lst[train_size+val_size:])
h5f.create_dataset('target_train', data=target_lst[0:train_size])
h5f.create_dataset('target_val', data=target_lst[train_size:train_size+val_size])
h5f.create_dataset('target_test', data=target_lst[train_size+val_size:])

h5f.close()
print("---done---")


model=combine_model(False,0)

final_train_gen = merge_generator(img_gen(img_dir_lst[0:train_size],batch_size_is), lstm_gen(feature_lst[0:train_size],target_lst[0:train_size], batch_size_is))
final_val_gen = merge_generator(img_gen(img_dir_lst[train_size:train_size+val_size],batch_size_is), lstm_gen(feature_lst[train_size:train_size+val_size],target_lst[train_size:train_size+val_size], batch_size_is))


print('debug1')
filepath="weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_sme', verbose=1, save_best_only=True, mode='max')
history=model.fit_generator(final_train_gen,  # Features, labels
                            steps_per_epoch=train_size//batch_size_is,
                            epochs=epochs,
                            validation_data=final_val_gen,
                            validation_steps=val_size // batch_size_is
                            callbacks=[checkpoint]
                            #verbose=1 
                            #callbacks=callbacks
                            )  


print('debug2')


# save training history
with open('my_model', 'wb') as file_1:
    pickle.dump(history.history, file_1)
#save model
model.save_weights( './my_model_weights.h5')
print('model saved')