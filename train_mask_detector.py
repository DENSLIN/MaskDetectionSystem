# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BatchSize = 96

DIRECTORY = r"dataset"
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("loading images...")

#image will be stored in array format in data
data = []
#image discribtion will be stored  in label
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
		# image in image fromat is loaded
    	image = load_img(img_path, target_size=(224, 224))
		#image is convrtd in array format
    	image = img_to_array(image)
    	image = preprocess_input(image)
		#data is pushed in allocated dataset
    	data.append(image)
    	labels.append(category)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
#converts label to digital value 
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#converting list into numpy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

#spliting data to train and test data
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=42)



# load the MobileNetV2 network and parameters of img in X,Y,layers 
inputStruc = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

# construct the structure network of the model 
Modelstuc = inputStruc.output
Modelstuc = AveragePooling2D(pool_size=(7, 7))(Modelstuc)
#flatten the whole image of 128  
flattenModel = Flatten(name="flatten")(Modelstuc)
flattenModel = Dense(128, activation="relu")(flattenModel)
#output layer where the model bring to output which is 2 
OutputLayer = Dropout(0.5)(flattenModel)
OutputLayer = Dense(2, activation="softmax")(OutputLayer)

# create the actual model 
model = Model(inputs=inputStruc.input, outputs=OutputLayer)

# loop over all layers in the base model and freeze them so they will
# not be updated during the first training process
for layer in inputStruc.layers:
	layer.trainable = False

# compile our model
print("compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# constructing more images by changing parametres to train and test modre data  
dataGenerator = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# train the head of the network
print(" training Model")
H = model.fit(
	# loading train data 
	dataGenerator.flow(trainX, trainY, batch_size=BatchSize),
	steps_per_epoch=len(trainX) // BatchSize,
	# loading test data
	validation_data=(testX, testY),
	validation_steps=len(testX) // BatchSize,
	epochs=EPOCHS)

# make predictions on the testing set
print("Testing  network")
predIdxs = model.predict(testX, batch_size=BatchSize)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

#classification report
print(classification_report(testY.argmax(axis=1), predIdxs,target_names=lb.classes_))

# Saving model 
print("[INFO] saving mask detector model")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")