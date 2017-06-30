import pickle

try:
    with open('mlph_features.dat', 'rb') as f:
        features = pickle.load(f)
        print(features)

except:
    print("Extracting features again...")
    try:
        import feature
    except:
        print("An exception occured!!")


