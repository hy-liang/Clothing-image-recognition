from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from VGG16_keras import VGG_16
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

model_weights_path = './data/CNN_data/vgg16_weights.h5'

def get_CNN_Category_model():
    # -------load-VGG16 network--------
    VGG_model = VGG_16()
    VGG_model.load_weights(model_weights_path)
    for layer in VGG_model.layers:
        layer.trainable = False
    # get convolution feature
    feature = VGG_model.layers[32].output
    # add an fullconnect layer
    Ds_hidden1 = Dense(2048, activation='relu')(feature)
    # add a drop_out layer
    Drop_out = Dropout(0.5)(Ds_hidden1)
    # add an fullconnect layer
    Ds_prediction = Dense(8, activation='softmax')(Drop_out)

    model = Model(input=VGG_model.input, output=Ds_prediction)
   #model.load_weights('./data/CNN_data/CNN_Category_model')

    return model

CNN_Category_model = get_CNN_Category_model()
#set parameters for compile
sgd = SGD(lr=0.001, momentum=0.95, decay=1e-6, nesterov=True)
CNN_Category_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(samplewise_center=True, rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(
    directory='./data/category_train',
    target_size=(224, 224),
    batch_size=50
)


hfit = CNN_Category_model.fit_generator(
    train_generator,
    samples_per_epoch=130000,
    nb_epoch=2
)
print hfit.history

CNN_Category_model.save_weights('./data/CNN_data/CNN_Category_model')

val_datagen = ImageDataGenerator()

val_generator = val_datagen.flow_from_directory(
    directory='./data/category_val/',
    target_size=(224, 224)
)

print CNN_Category_model.evaluate_generator(val_generator ,val_samples=2000)
