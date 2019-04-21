from keras.utils import np_utils
import numpy as np
import sys,os
from hog_feature import cal_hog_feature
from color_feature import cal_hist_feature
from Small_Sample_CNN_model import Small_Sample_CNN
from keras.models import Model
from sklearn import preprocessing
#os.chdir(os.path.split(os.path.abspath(__file__))[0])
#print 'load data cwd:', os.getcwd()
category_name = { 0:'Dress', 1:'Sweater', 2:'Tank', 3:'Tee', 4:'Joggers', 5:'Jeans', 6:'Shorts', 7:'Skirt'}
class_of_usage = {'Dress':0, 'Sweater':1, 'Tank':1, 'Tee':1, 'Joggers':2, 'Jeans':2, 'Shorts':2, 'Skirt':2}
class_of_usage_name = {0:u'Dress', 1:'Tops', 2:'Trousers'}


'''''
Small_X_train----50*50picture of train set, shape = (sample_num, 50, 50 ,3)
Small_Y_train----8 category target of train set1D
Small_X_val------50*50picture of val set, shape = (sample_num, 50, 50 ,3)
Small_Y_val------8 category target of val set,1D
Y_train----------8 category target of train set,2D
Y_val----------8 category target of val set,2D
Y_train_usage----3 class of usage target of train set,2D
Y_val_usage------3 class of usage target of val set,2D
hog_X_train-----hog feature of train data set
hog_X_val------hog feature of val data set
hist_X_train----hist color feature of train data set
hist_X_val----hist color feature of val data set
'''''

def get_usage_target():
    train_usage_path = './data/Y_train_usage.npy'
    val_usage_path = './data/y_val_usage.npy'
    y_train = []
    y_val = []
    if os.path.exists(train_usage_path):
        y_train = np.load(train_usage_path)
    else:
        for t in Small_Y_train:
            y_train.append(class_of_usage[category_name[t]])
        y_train = np.array(y_train)
        y_train = np_utils.to_categorical(y_train, 3)
        np.save(train_usage_path, y_train)

    if os.path.exists(val_usage_path):
        y_val = np.load(val_usage_path)
    else:
        for t in Small_Y_val:
            y_val.append(class_of_usage[category_name[t]])
        y_val = np.array(y_val)
        y_val = np_utils.to_categorical(y_val, 3)
        np.save(val_usage_path, y_val)
    return y_train, y_val

def load_small_data():
    small_X_train = np.load('./data/Small_X_train.npy')
    small_Y_train = np.load('./data/Small_Y_train.npy')
    small_X_val = np.load('./data/Small_X_val.npy')
    small_Y_val = np.load('./data/Small_Y_val.npy')
    return small_X_train, small_Y_train, small_X_val, small_Y_val

def get_Y():
    if os.path.exists('./data/Y_train.npy'):
        y_train = np.load('./data/Y_train.npy')
    else:
        y_train = np_utils.to_categorical(Small_Y_train, 8)
        np.save('./data/Y_train.npy', y_train)

    if os.path.exists('./data/Y_val.npy'):
        y_val = np.load('./data/Y_val.npy')
    else:
        y_val = np_utils.to_categorical(Small_Y_val, 8)
        np.save('./data/Y_val.npy', y_val)

    return y_train,y_val

def get_hog_feature():
    train_path = './data/hog_feature_train.npy'
    val_path = './data/hog_feature_val.npy'
    if os.path.exists(train_path) and os.path.exists(val_path):
        x_train = np.load('./data/hog_feature_train.npy')
        x_val = np.load('./data/hog_feature_val.npy')
    else:
        x_train,x_val = cal_hog_feature(img_X_train, img_X_val)

    #x_train  = preprocessing.scale(x_train)
    #x_val = preprocessing.scale(x_val)

    return x_train, x_val

def get_hist_feature():
    train_path = './data/hist_feature_train.npy'
    val_path = './data/hist_feature_val.npy'
    if os.path.exists(train_path) and os.path.exists(val_path):
        x_train = np.load('./data/hist_feature_train.npy')
        x_val = np.load('./data/hist_feature_val.npy')
    else:
        x_train,x_val = cal_hist_feature(img_X_train, img_X_val)

    #x_train  = preprocessing.scale(x_train)
    #x_val = preprocessing.scale(x_val)

    return x_train, x_val

def cal_CNN_feature():
    CNN_Category_model = Small_Sample_CNN()
    CNN_Category_model.load_weights('./data/Small_Sample_CNN_model')
    print('getting cnn feature!')
    model = Model(inputs=CNN_Category_model.input, outputs=CNN_Category_model.layers[14].output)
    x_train = model.predict(Small_X_train)
    x_val = model.predict(Small_X_val)
    np.save('./data/cnn_feature_train.npy', x_train)
    np.save('./data/cnn_feature_val.npy', x_val)

def get_cnn_feature():
    train_path = './data/cnn_feature_train.npy'
    val_path = './data/cnn_feature_val.npy'
    if os.path.exists(train_path) and os.path.exists(val_path):
        x_train = np.load('./data/cnn_feature_train.npy')
        x_val = np.load('./data/cnn_feature_val.npy')
    else:
        cal_CNN_feature()
        x_train = np.load('./data/cnn_feature_train.npy')
        x_val = np.load('./data/cnn_feature_val.npy')
    #x_train = preprocessing.scale(x_train)
    #x_val = preprocessing.scale(x_val)
    return x_train, x_val


img_X_train, Small_Y_train, img_X_val, Small_Y_val = load_small_data()

# convert class vectors to binary class matrices
Y_train, Y_val = get_Y()

Small_X_train = img_X_train.astype('float32')
Small_X_val = img_X_val.astype('float32')
Small_X_train /= 255.0
Small_X_val /= 255.0

'''''
Y_train_usage, Y_val_usage = get_usage_target()

hog_X_train, hog_X_val = get_hog_feature()

hist_X_train, hist_X_val = get_hist_feature()

cnn_X_train, cnn_X_val = get_cnn_feature()


late_fusion_cnn_hog_train = np.load('./data/late_fusion_cnn_hog_train.npy')
late_fusion_cnn_hog_val = np.load('./data/late_fusion_cnn_hog_val.npy')
late_fusion_hog_hist_train = np.load('./data/late_fusion_hog_hist_train.npy')
late_fusion_hog_hist_val = np.load('./data/late_fusion_hog_hist_val.npy')
late_fusion_cnn_hist_train = np.load('./data/late_fusion_cnn_hist_train.npy')
late_fusion_cnn_hist_val = np.load('./data/late_fusion_cnn_hist_val.npy')
late_fusion_cnn_hog_hist_train = np.load('./data/late_fusion_cnn_hog_hist_train.npy')
late_fusion_cnn_hog_hist_val = np.load('./data/late_fusion_cnn_hog_hist_val.npy')
'''''
print('X_train shape:', Small_X_train.shape)
print('Y_train shape:', Y_train.shape)
print(Small_X_train.shape[0], 'train samples')
print(Small_X_val.shape[0], 'test samples')
