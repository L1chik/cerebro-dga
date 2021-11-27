import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics


def load_data():
    data = {'site': [], 'tag': []}
    how = 0
    for line in open('datasets/testmultiv2.txt'):
        how += 1
        if how == 70000:
            return data
        row = line.split("#")
        data['site'] += [row[0]]
        data['tag'] += [row[1]]
    return data


def train_test_split(data, validation_split=0.1):
    sz = len(data['site'])
    indicates = np.arange(sz)
    np.random.shuffle(indicates)
    X = [data['site'][i] for i in indicates]
    Y = [data['tag'][i] for i in indicates]
    nb_validation_samples = int(validation_split * sz)
    return {
        'train': {'x': X[:-nb_validation_samples], 'y': Y[:-nb_validation_samples]},
        'test': {'x': X[:-nb_validation_samples], 'y': Y[:-nb_validation_samples]}
    }


def train():
    data = load_data()

    d = train_test_split(data)

    text_clf = Pipeline([(('tlidf'), TfidfVectorizer()), ('clf', SGDClassifier(loss='modified_huber')), ])

    text_clf.fit(d['train']['x'], d['train']['y'])

    predicted = text_clf.predict(d['test']['x'])

    filename = 'modelV2021isp_multi2.sav'
    pickle.dump(text_clf, open(filename, 'wb'))
    print(metrics.classification_report(digits=6, y_true=d['test']['y'], y_pred=predicted))
    print(metrics.confusion_matrix(y_true=d['test']['y'], y_pred=predicted))
    print("accuracy_score")
    print(metrics.accuracy_score(y_true=d['test']['y'], y_pred=predicted))
    y_prob = text_clf.predict_proba(d['test']['x'])
    print(metrics.roc_auc_score(d['train']['y'], y_prob, multi_class='ovo'))
    print(y_prob)


if __name__ == '__main__':
    sys.exit(train())
