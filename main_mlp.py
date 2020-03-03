import numpy as np
from constants import N_FOLDS
from model import MyMultilayerPerceptron
from sklearn.metrics import accuracy_score
from train import train


for i in range(N_FOLDS):
    current_fold = str(i + 1)
    train_fn = '[YOUR TRAIN DATA FILE PATH HERE]'
    validation_fn = '[YOUR VALIDATION DATA FILE PATH HERE]'
    test_fn = '[YOUR TEST DATA FILE PATH HERE]'
    target_train_fn = '[YOUR TRAIN LABELS FILE PATH HERE]'
    target_validation_fn = '[YOUR VALIDATION LABELS FILE PATH HERE]'
    target_test_fn = '[YOUR TEST LABELS FILE PATH HERE]'

    # loading the data already splitted
    x_train = np.load(train_fn)
    x_validation = np.load(validation_fn)
    x_test = np.load(test_fn)
    y_train = np.load(target_train_fn)
    y_validation = np.load(target_validation_fn)
    y_test = np.load(target_test_fn)
    n_classes = 12
    model = MyMultilayerPerceptron(n_classes, units=512, dropout_rate=0.5)
    print("Fold %s metrics:\n" % current_fold)
    # TRAINING
    train(model, x_train, y_train, x_validation, y_validation)

    # TESTING
    pred = model.predict(x_test)
    print(accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))