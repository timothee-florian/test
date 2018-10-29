import numpy as np

def evaluation_glob(final,y_te):
    """"gives the proportion of correct predictions"""
    sum=0
    for j in range (0, len(final)):
        if (y_te[j]==final[j]):
            sum =sum +1
    return (sum/(len(y_te)))

def evaluation(final,y_te):
    """"evaluation of the proportion of boson (-) that are predicted as boson(-1) and other particles (+1) 
    that are predicted as other particles"""
    real_pos =0
    real_neg =0
    pos=0
    neg=0
    for j in range (0, len(final)):
        if (y_te[j]==1):
            real_pos = real_pos + 1
            if (y_te[j]==final[j]):
                pos = pos +1
        else:
            real_neg = real_neg + 1
            if (y_te[j]==final[j]):
                neg = neg +1
        
    return (pos/real_pos, neg/real_neg)

def evaluation_pos(final,y_te):
    """"evaluation of the proportion of bosons (-1) 
    that are predicted as bosons"""
    real_pos =0
    real_neg =0
    pos=0
    neg=0
    for j in range (0, len(final)):
        if (y_te[j]==1):
            real_pos = real_pos + 1
            if (y_te[j]==final[j]):
                pos = pos +1
        else:
            real_neg = real_neg + 1
            if (y_te[j]==final[j]):
                neg = neg +1
        
    return pos/real_pos


def evaluation_neg(final,y_te):
    """"evaluation of the proportion of bosons (-1) 
    that are predicted as bosons"""
    real_pos =0
    real_neg =0
    pos=0
    neg=0
    for j in range (0, len(final)):
        if (y_te[j]==1):
            real_pos = real_pos + 1
            if (y_te[j]==final[j]):
                pos = pos +1
        else:
            real_neg = real_neg + 1
            if (y_te[j]==final[j]):
                neg = neg +1
        
    return neg/real_neg



def evaluate_the_predictions_moins(les_pred):
    """for 3 results of prediction from polynomial regression, this function return the most dominant prediction"""
    final = []
    for i in range (0,len(les_pred[0])):
        pos = 0
        neg = 0
        for vect in les_pred:
            if(vect[i]==-1):
                neg = neg + 1
            else:
                pos = pos + 1
            
        if(pos>1):
            final.append(1)
        else:
            final.append(-1)
    return final


        
