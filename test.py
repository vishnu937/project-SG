from svmutil import *
import numpy as np
import matplotlib.image


try:
    test_data = np.load("test_data_by_numpy.npy")
    print('test data size is:', test_data.shape)

except:
    print("Extracting test_data features again...")
    try:
        import get_test_data
    except:
        print("An exception occured!!")

m = svm_load_model('SVM_test_model')
print(m)
svm_type = m.get_svm_type()
nr_class = m.get_nr_class()
# svr_probability = m.get_svr_probability()
class_labels = m.get_labels()
# sv_indices = m.get_sv_indices()
nr_sv = m.get_nr_sv()
is_prob_model = m.is_probability_model()
# support_vector_coefficients = m.get_sv_coef()
# support_vectors = m.get_SV()


print('SVM Type = ', svm_type)
print('Number of classes = ', nr_class)
# print('SVM model probability', svr_probability)
print('Class labels = ', class_labels)
# print('SV indices = ', sv_indices)
print('Totel SV = ', nr_sv)
print('is probability model = ', is_prob_model)
# print('SV coefficients = ', support_vector_coefficients)
# print('Support Vectors', support_vectors)
# print(len(test_data.tolist()))

try:
    np_labels = np.load('predicted_labels.npy')
except:
    y = [0]*len(test_data.tolist())
    x = test_data.tolist()
    p_labs, p_acc, p_vals = svm_predict(y, x, m)
    np_labels = np.array(p_labs)
    np.save('predicted_labels', np_labels)


# print('list of predicted labels = ', p_labs)
# print('classification accuracy  ,  mean squared error,   squared correlation coefficients ', p_acc)
# print('list of decision values or probability estimates', p_vals)
print('predicted labels size = ', np_labels.shape)
# print(np_labels[0:10])
resize_label = np.resize(np_labels, (496, 2251))
print('resized shape = ', resize_label.shape)

padded_labels = np.lib.pad(resize_label, 2, 'edge')
padded_labels = padded_labels.astype(int)
print('shape after padding = ', padded_labels.shape)

# plt.imshow(padded_labels)
# plt.show()
h, w = padded_labels.shape
rgb = np.zeros((h, w, 3))
list_labels = np.unique(padded_labels)
print('predicted labels = ', list_labels)
for x in range(0, h):
        for y in range(0, w):

            if padded_labels[x, y] == 1:   # water
                rgb[x, y, 2] = 255   # blue
            elif padded_labels[x, y] == 2:   # flood plane
                rgb[x, y, 0] = 255   # yellow
                rgb[x, y, 1] = 255
            elif padded_labels[x, y] == 3:   # irrigation
                rgb[x, y, 0] = 255    # Red
            elif padded_labels[x, y] == 4:  # vegetation
                rgb[x, y, 1] = 255   # Green
            elif padded_labels[x, y] == 5:   # urban # white
                rgb[x, y, 0:3] = 255
                # rgb[x, y, 2] = 255
            else:
                print('error more than 5 classes')
                # rgb[x, y, 0:3] = 0

matplotlib.image.imsave('test.png', rgb.astype(np.uint8))
