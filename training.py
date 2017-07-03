import pickle
from svmutil import *
# import numpy as np

try:
    with open('train_features.dat', 'rb') as f:
        train_data = pickle.load(f)
        print('feature size is:', train_data.shape)
        # print(train_data)
    with open('target_class.dat', 'rb') as f:
        target_class = pickle.load(f)
        # print(target_class)
        print('true class label size', target_class.shape)

except:
    print("Extracting features again...")
    try:
        import feature
    except:
        print("An exception occured!!")


# print(train_data[0:10].tolist())
# parm = svm_parameter(kernel_type = LINEAR, C = 10)
# -s 0 C-SVM (multiclass classification) - default
parm = svm_parameter('-s 0 -t 0 -c 10')
prob = svm_problem(target_class.tolist(), train_data.tolist())
# parm.cross_validation = 1
# parm.nr_fold = 10
m = svm_train(prob, parm)

svm_save_model('SVM_test_model', m)

