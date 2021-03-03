import os
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
from skimage import feature
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(63)

flowers={0:'daffodil', 1:'lily', 2:'rose', 3:'sunflower'}

label=0
data = []
for folder in os.listdir(path='input/flowers'):
    for subfolder in os.listdir(path='input/flowers/'+folder):
        img_dir='input/flowers/'+folder+'/'+subfolder
        img=cv2.imread(img_dir)
        img = cv2.resize(img, (128, 256))
        hog_image = feature.hog(img, orientations=9,
                                pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2),
                                block_norm='L2-Hys', transform_sqrt=True)
        data.append([hog_image,label])
    label+=1

np.random.shuffle(data)
df=pd.DataFrame()
X=np.array([e[0] for e in data])
y=np.array([e[1] for e in data])

svm=LinearSVC(random_state=63, tol=1e-05)
svm.fit(X,y)

label=0
test = []
for file in os.listdir(path='test_images/flowers'):
    img_dir='test_images/flowers/'+file
    img=cv2.imread(img_dir)
    img = cv2.resize(img, (128, 256))
    hog_image = feature.hog(img, orientations=9,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),
                            block_norm='L2-Hys', transform_sqrt=True)
    test.append(hog_image)

y_true=[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
y_preds=svm.predict(test)

print('accuracy: {:g}%'.format(accuracy_score(y_true, y_preds)*100))

plt.close()
cm=confusion_matrix(y_true,y_preds)
sns.heatmap(cm/cm.sum(axis=1), annot=True, fmt='0.2f')
plt.show()







