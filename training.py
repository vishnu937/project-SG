import pickle
from svmutil import *

try:
    with open('train_features.dat', 'rb') as f:
        train_data = pickle.load(f)
        print('feature size is:', train_data.shape)
        print(train_data)
    with open('target_class.dat', 'rb') as f:
        target_class = pickle.load(f)
        print(target_class)

except:
    print("Extracting features again...")
    try:
        import feature
    except:
        print("An exception occured!!")



