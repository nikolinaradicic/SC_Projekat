import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.models import load_model

img_width, img_height = 256, 256
validation_data_dir = "C:\\Users\\UROS_PC\\Desktop\\instalabeled"
test_data_dir = ""
nb_train_samples = 1000
nb_validation_samples = 450
batch_size = 20

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    batch_size= 20,
    target_size=(img_height, img_width),
    shuffle=False,
    class_mode="categorical")

model = load_model('C:\\Users\\UROS_PC\\Devgg_ft.h5')
fnames = validation_generator.filenames
ground_truth = validation_generator.classes

label2index = validation_generator.class_indices
idx2label = dict((v, k) for k, v in label2index.items())
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
errors = []
for i in range(0, nb_validation_samples):
    if predicted_classes[i] != ground_truth[i]:
        errors.append(i)
print("No of errors = {}/{}".format(len(errors),nb_validation_samples))

for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]

    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])

    print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class]))

    original = load_img('{}/{}'.format(validation_data_dir, fnames[errors[i]]))
    plt.title(title)
    plt.imshow(original)
    plt.show()