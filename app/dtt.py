import h5py
import time
import numpy as np
from joblib import dump, load

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

CLSF = ['svm', 'knn', 'randomforest']

def test(h5_file_name, joblib_model_file_name):
    print("==============  Data Testing  ===============")
    model = load(joblib_model_file_name)

    print(f"Predict all data from {h5_file_name} ...")
    tic = time.perf_counter()
    hf = h5py.File(h5_file_name, 'r')
    labels = []
    labels_unique = []
    preds = []
    for key in hf.keys():
        group = hf.get(key)
        labels_unique.append(key)
        for index in range(0, len(group.items())):
            data = f"{key}{index}"
            labels.append(key)
            h5_hog_desc = group.get(data)
            hog_desc = np.array(h5_hog_desc)
            pred = model.predict(hog_desc.reshape(1, -1))[0]
            preds.append(pred)
            print(f"predict {data} as {pred} result {pred == key}")

    toc = time.perf_counter()
    elapsed = toc - tic
    print(f"Model   : {joblib_model_file_name}")
    print(f"Data    : {h5_file_name}")
    print(f"Elapsed : {elapsed} sec")
    print("=========      Classification Report      =========")
    print(classification_report(labels, preds, target_names=labels_unique))
    print(f"ConfMatrx:\n{confusion_matrix(labels, preds, labels=labels_unique)}")

    

def train(h5_file_name, classification, attributes):
    print("==============  Data Training  ==============")

    print("Checking attributes for classification ...")
    C = 0.0
    kernel = 'rbf'
    tol = 0
    coef0 = 0.0

    neighbors = 0
    depth = 0

    rstate = 0
    if (classification == CLSF[0]):
        C = float(attributes[0].split('=')[1])
        kernel = attributes[1].split('=')[1]
        tol = float(attributes[2].split('=')[1])
        if (len(attributes) > 3):
            coef0 = float(attributes[3].split('=')[1])
    elif (classification == CLSF[1]):
        neighbors = int(attributes[0].split('=')[1])
    elif (classification == CLSF[2]):
        depth = int(attributes[0].split('=')[1])
        rstate = int(attributes[1].split('=')[1])
    else:
        print("unsupported classification")
        return
    print("Checking attributes for classification done!")

    hog_descs = []
    labels = []

    print(f"Reading all data from {h5_file_name} ...")
    hf = h5py.File(h5_file_name, 'r')
    for key in hf.keys():
        group = hf.get(key)
        for index in range(0, len(group.items())):
            data = f"{key}{index}"
            labels.append(key)
            h5_hog_desc = group.get(data)
            hog_desc = np.array(h5_hog_desc)
            hog_descs.append(hog_desc)
            print(f"read {data}")
    print(f"Reading all data from {h5_file_name} done!")

    if (classification == CLSF[0]):
        print(f"Training with SVM (C={C}, kernel={kernel}, tol={tol}, coef0={coef0}) ...")
        svm_model = SVC(C=C, kernel=kernel, tol=tol, coef0=coef0)
        tic = time.perf_counter()
        svm_model.fit(hog_descs, labels)
        toc = time.perf_counter()
        elapsed = toc - tic
        dump(svm_model, f"model-svm-C{C}-{kernel}-tol{tol}-coef{coef0}-{h5_file_name.split('.')[0]}.joblib")
        print(f"Training with SVM (C={C}, kernel={kernel}, tol={tol}, coef0={coef0}) done in {elapsed}s!")
    elif (classification == CLSF[1]):
        print(f"Training with KNN (n={neighbors}) ...")
        knn_model = KNeighborsClassifier(n_neighbors=neighbors)
        tic = time.perf_counter()
        knn_model.fit(hog_descs, labels)
        toc = time.perf_counter()
        elapsed = toc - tic
        dump(knn_model, f"model-knn-n{neighbors}-{h5_file_name.split('.')[0]}.joblib")
        print(f"Training with KNN (n={neighbors}) done in {elapsed}s!")
    elif (classification == CLSF[2]):
        print(f"Training with Random Forest (depth={depth}, random state={rstate}) ...")
        rf_model = RandomForestClassifier(max_depth=depth, random_state=rstate)
        tic = time.perf_counter()
        rf_model.fit(hog_descs, labels)
        toc = time.perf_counter()
        elapsed = toc - tic
        dump(rf_model, f"model-rf-d{depth}-rs{rstate}-{h5_file_name.split('.')[0]}.joblib")
        print(f"Training with Random Forest (depth={depth}, random state={rstate}) done in {elapsed}s!")
    else:
        print("unsupported classification")
