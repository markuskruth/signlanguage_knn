import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

def load_data():
	imagepath = "Sign-Language-Digits-Dataset/Dataset"
	digit_folders = ["0","1","2","3","4","5","6","7","8","9"]

	X = [] # features
	y = [] # labels

	# import the image-data
	for folder in digit_folders:
		for filename in glob.glob(imagepath + "/"+folder + "/*.jpg"):
			img = Image.open(filename)
			grayscale_img = ImageOps.grayscale(img)
			npImg = np.asarray(grayscale_img)

			X.append(npImg)
			y.append(int(folder))

	X = np.asarray(X).reshape(len(X), 100*100) # transform X into a numpy array and reshape the images

	return X,y


def pca_init(X, N):
	pca = PCA(n_components=N)
	pca.fit(X)

	return pca


#X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, test_size=0.3, random_state=11)

def train_model(X, K):
	clf = KNeighborsClassifier(n_neighbors=K)
	scores = cross_val_score(clf, X, y, cv=10)
	print("Model accuracy:", sum(scores)/10)

	clf.fit(X, y)

	return clf


#y_pred = clf.predict(X_val)
#multi_accuracy = accuracy_score(y_val, y_pred)

def generate_confusion_matrix(y_true, y_pred):
    # visualize the confusion matrix
    ax = plt.subplot()
    c_mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(c_mat, annot=True, fmt='g', ax=ax)

    ax.set_xlabel('Predicted labels', fontsize=15)
    ax.set_ylabel('True labels', fontsize=15)
    ax.set_title('Confusion Matrix', fontsize=15)

#print(multi_accuracy)
#generate_confusion_matrix(y_val, y_pred)
#plt.show()

N = 100
if __name__ == "__main__":
	X,y = load_data()
	pca = pca_init(X, N)
	X_reduced = pca.transform(X)

	clf = train_model(X_reduced, K=10)