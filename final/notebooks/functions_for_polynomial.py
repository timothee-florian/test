import numpy as np
from functions import *
from combinatory import *
from evaluation_function import *
from proj1_helpers import *
import matplotlib.pyplot as plt

def build_indices_ratio(y, ratio, seed):
    """build indices for splitting the original train sample"""
    nb_row = y.shape[0]
    np.random.seed(seed)
    indices = np.random.permutation(nb_row)
    nb_split = int(nb_row*ratio)
    ind_train = indices[nb_split:]
    ind_test = indices[:nb_split]
    return ind_train, ind_test

def polynomial_regressions(poly,y):
    # least squares applied to polynom
    w = ridge_regression(y,poly, 0)
    return w

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree"""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def compute_losses_pos_neg(y, tx, w):
    """Calculate the loss MSE separately for bosons and the other particles"""
    e = y - np.dot(tx, w)
    e_pos = e[y>0]
    e_neg = e[y<0]
    pos= len(e_pos)
    neg= len(e_neg)
    return np.sum(e_pos * e_pos)/ (2 * pos),np.sum(e_neg * e_neg)/ (2 *neg)

def bias_variance_demo_poly(x_c,label,ratio,degrees):
    """Bias variance decomposition, for the values of a given feature we want to 
    to find the polynom (until degree 20) that allows to get the best prediction"""
    # define parameters

    seeds = range(10)
    
    adapted_degree=[]
    
    # define list to store the variable
    
    rmse_tr = np.empty((len(seeds), len(degrees)))
    rmse_te = np.empty((len(seeds), len(degrees))) 
    
    for index_seed, seed in enumerate(seeds):
    
        ind_train, ind_test = build_indices_ratio(label, ratio, seed)
    
        y_train = label[ind_train].reshape(-1,1)
        x_train =x_c[ind_train]
        
        y_test = label[ind_test].reshape(-1,1)
        x_test =x_c[ind_test]
       
        rmse_tr_p =[]
        rmse_tr_n =[]
        rmse_te_p =[]
        rmse_te_n =[]
        for index_degree, degree in enumerate(degrees):
                poly_train = build_poly(x_train, degree)
                poly_test = build_poly(x_test, degree)
    
                w= polynomial_regressions(poly_train,y_train)
       
                loss_tr_p, loss_tr_neg = compute_losses_pos_neg(y_train, poly_train, w)
            
                loss_te_p,loss_te_neg = compute_losses_pos_neg(y_test, poly_test, w)
                
                rmse_tr_p.append(loss_tr_p) 
                rmse_tr_n.append(loss_tr_neg)
            
                rmse_te_p.append(loss_te_p) 
                rmse_te_n.append(loss_te_neg)
           
                  
                rmse_tr_p=normalize_vect(rmse_tr_p)
                rmse_tr_n=normalize_vect(rmse_tr_n)
            
                rmse_te_p=normalize_vect(rmse_te_p) 
                rmse_te_n=normalize_vect(rmse_te_n)
            
                rmse_tr_f = sum_vector(rmse_tr_p,rmse_tr_n)
                rmse_te_f = sum_vector(rmse_te_p,rmse_te_n)
        
        for j in range (0,len(rmse_tr_p)):
            rmse_tr[index_seed,j] = rmse_tr_f[j]
            rmse_te[index_seed,j] = rmse_te_f[j]
    
    return rmse_tr,rmse_te

def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")
    
def normalize_vect(a):
    "normalize a vector with the max"
    maxi = max(a)
    for i in range (0,len(a)):
        a[i]=a[i]/maxi
    return a

def sum_vector(a,b):
    "sum and averages two vectors termby term"
    c=[]
    for i in range (0,len(a)):
        c.append(a[i]+b[i]/2)
    return c
def best_degree_of_polynom(rmse_tr,rmse_te,degrees):
    """after the biased variance decomposition, this function do the average of the errors for the train and the test
    and it returns the degree for which the test error is minimised"""
    
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    
    list_rmse_tr_mean=rmse_tr_mean.tolist()
    list_rmse_te_mean=rmse_te_mean.tolist()
    
    return degrees[list_rmse_te_mean[0].index(min(list_rmse_te_mean[0]))]

def release_best_degree_by_feature(x,y,ratio,degrees):
    "return the list of degree of polynom that allows the best polynomial regression for all the 30 fetaures"
    adapted_degree=[]
    for j in range (0,30):
        rmse_tr,rmse_te = bias_variance_demo_poly(x[:,j],y,ratio,degrees)
        best_degree=best_degree_of_polynom(rmse_tr,rmse_te,degrees)
        adapted_degree.append(best_degree)
        print("for feature: ",j,"the best degree is:,",best_degree)
    return adapted_degree

def poly_regression_with_adapted_degree(x,y,num,adapted_degree,ratio):
    """returns the predictions for the feature applied in argument (num=[0,1], means we want to test features 0 and 1).
    polynomial regresion is applied on these features with the best degree for each feature, this degree was determined 
    with the function release_best_degree_by_feature
    """
    
    les_pred = []
    seed =1
    
    ind_train, ind_test = build_indices_ratio(y, ratio, seed)
    y_train = y[ind_train]
    x_train =x[ind_train]
        
    y_test = y[ind_test]
    x_test =x[ind_test]

    
    for i in num:
        poly_train = build_poly(x_train[:,i], adapted_degree[i])
        w= polynomial_regressions(poly_train,y_train)

        poly_test = build_poly(x_test[:,i], adapted_degree[i])
        y_pred = predict_labels(w, poly_test)
    
        les_pred.append(y_pred)
    return les_pred,y_test

def final_poly_regression_with_adapted_degree(x,y,test,num,adapted_degree):
    """returns the predictions for the feature applied in argument (num=[0,1], means we want to test features 0 and 1).
    polynomial regresion is applied on these features with the best degree for each feature, this degree was determined 
    with the function release_best_degree_by_feature
    """  
    les_pred = []
    
    for i in num:
        poly = build_poly(x[:,i], adapted_degree[i])
        w= polynomial_regressions(poly,y)
        poly_test = build_poly(test[:,i], adapted_degree[i])
        y_pred = predict_labels(w, poly_test)
    
        les_pred.append(y_pred)
    return les_pred


def evaluation_all_polynomial_regression(x,y,adapted_degree):
    ratio =0.7
    the_best_feature=[]
    print("feature",",","Boson (-1)",",","Others (+1)",",","Global evaluation")
    for j in range (0,30):
        num =[j]
        les_pred,y_test = poly_regression_with_adapted_degree(x,y,num,adapted_degree,ratio)
        prediction=les_pred[0]
        neg=evaluation_neg(prediction,y_test)
        pos =pos=evaluation_pos(prediction,y_test)
        print("feature",j,",",neg,",",pos,",",evaluation_glob(prediction,y_test))    
        if (pos>0.3):
            the_best_feature.append(j)
    return the_best_feature

def test_combinations(the_best_features,x,y,adapted_degrees):
    """tests all combinations of three elements for features that gave the best result for polynomial regression
    and returns the best combination"""
    list_combi =combinationsUniques(the_best_features,3)
    best_result=0
    best_combi=[]
    for num in list_combi:
        ratio=0.7
        les_pred,y_test =poly_regression_with_adapted_degree(x,y,num,adapted_degrees,ratio)
        final=evaluate_the_predictions_moins(les_pred)
        print(num)
        print(evaluation(final,y_test))
        glob=evaluation_glob(final,y_test)
        print(glob)
        if (glob>best_result):
            best_result=glob
            best_combi=num
    return best_combi

def final_prediction(x,y,test,best_combi,list_degrees):
    les_pred=final_poly_regression_with_adapted_degree(x,y,test,best_combi,list_degrees)
    final=evaluate_the_predictions_moins(les_pred)
    return final
        
        