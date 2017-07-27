from svmutil import *
import numpy as np

try:
    train_data = np.load('train_features.npy')
    target_class = np.load('target_class.npy')
    print('feature size is:', train_data.shape)
    print('true class label size', target_class.shape)
except:
    print("Extracting features again...")
    try:
        import feature
    except:
        print("An exception occured!!")


# sys.path.append('D:\python soft\libsvm-3.22\libsvm-3.22\python')

# print(train_data[0:10].tolist())
# parm = svm_parameter(kernel_type = LINEAR, C = 10)
# -s 0 C-SVM (multiclass classification) - default
parm = svm_parameter('-s 0 -t 0 -c 10 -h 0')
y = target_class.tolist()
x = train_data.tolist()
prob = svm_problem(y, x)
parm.cross_validation = 1
parm.nr_fold = 5
m = svm_train(prob, parm)

# svm_save_model('SVM_test_model', m)

