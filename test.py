from svmutil import *

m = svm_load_model('SVM_test_model')

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








