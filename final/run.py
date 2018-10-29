import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from functions import *
from implementations import *
y, x, ids = load_csv_data('train.csv')


columns_to_keep = [0,1,2,3,4,5,6,10,11,12,13,22,24]
x1 = x[:, columns_to_keep]
y1=y
#remove -999 and put them close to the rest of the data
min_ = x1[x1!=-999].min()-1
x1[x1<=-998]= min_

# remove outliers
x1 = x1
y1 = y1
threshold = []
for i in range(x1.shape[1]):
    # get value of each column of the 99 percentile
    threshold += [np.percentile(x1[:,i], 99)]
for i in range(x1.shape[1]):
    y1 = y1[x1[:,i]<=threshold[i]]
    x1 = x1[x1[:,i]<=threshold[i]]
# print('{} % of the data was removed'.format(int(100*(1-len(x1)/len(x)))))

degree =12
x_train, x_test, y_train, y_test = split_data(x1, y1, ratio=0.8, seed =2)
x_train, mean_x, std_x= standardize(x_train)

x_test = (x_test-mean_x)/std_x
x_train = build_poly(x_train, degree=degree)
x_test = build_poly(x_test, degree=degree)

w, loss = ridge_regression(y_train, x_train, lambda_=10**-7)
y_p = predict_labels(w, x_test)
accuracy = np.mean(y_p == y_test)

error_tr = compute_error(y_train, x_train, w, 'mse')
error_te = compute_error(y_test, x_test, w, 'mse')
# print('Train error: {}, \ntest error: {},\naccuracy: {}'.format(error_tr, error_te, accuracy))

y_t, x_t, ids = load_csv_data('test.csv')
x_t1 = x_t[:,columns_to_keep]
x_t1[x_t1==-999]= min_
x_t2 = (x_t1-mean_x)/std_x
x_t2 = build_poly(x_t2, degree=degree)
y_p = predict_labels(w, x_t2)

create_csv_submission(ids, y_p, 'prediction_ridge_regression.csv')