

from keras.models import load_model
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

vrai = 0
brain = load_model("C:\\Users\\pauvp\\Desktop\\IA\\Conseil_des_Mikes\\140_90_50.h5")
chat = 0
bons_chats = 0
bons_chiens = 0
images = glob("EXERCICE/*.jpg")
nb = len(images)
for image in tqdm(images):
	nom = image
	image = cv2.imread(image)
	image = cv2.resize(image, (100, 100))
	image = image.astype("float32")
	image = image.reshape((1, 30000))
	image /= 255

	image = np.array(image)
	prediction = brain.predict(image, verbose=False)
	prediction = np.argmax(prediction)
	if "cat" in nom:
		chat += 1
	if "cat" in nom and int(prediction) == 1:
		vrai += 1
		bons_chats += 1
	elif "dog" in nom and int(prediction) == 0:
		vrai += 1
		bons_chiens += 1

print("j'ai eu:", vrai, "sur un total de", str(nb), "images")
print("Il y a {} chats".format(chat))
print("J'ai {} bons pour les chats et {} bons pour les chiens".format(bons_chats, bons_chiens))
