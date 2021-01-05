import numpy as np
import random
import os
import pickle
import math
import seaborn as sns
import matplotlib.pyplot as plt

import copy

from definitions import *
################################## measurement ##########################################    
    
def f1_score(TP,FP,FN):
    return 2*TP / (2*TP + FP + FN)

def f1_scores(CM):
    
    output=[f1_score(CM[i,i],np.sum(CM[:,i])-CM[i,i],np.sum(CM[i])-CM[i,i]) for i in range(CM.shape[0])]
    return output
    
def accuracy_scores(CM):
    
    output=[CM[i,i]/np.sum(CM[i]) for i in range(CM.shape[0])]
    
    return output


################################## venn diagramme ########################################## 
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles

def venn_2counts(a,b):
    
    ab=np.intersect1d(a, b)
    
    ab=len(ab)
    a_minusb=len(a)-ab
    
    b_minusa=len(b)-ab
    
    return a_minusb, b_minusa,ab
def venn_3counts(a,b,c):
    
    ab=np.intersect1d(a, b)
    abc=np.intersect1d(ab, c)
    
    abc_len=len(abc)
    ab_minusc=len(ab)-abc_len
    
    bc=np.intersect1d(b, c)    
    bc_minusa=len(bc)-abc_len
    
    ac=np.intersect1d(a, c) 
    ac_minusb=len(ac)-abc_len
    
    solo_a=len(a)-(ab_minusc+abc_len+ac_minusb)
    solo_b=len(b)-(ab_minusc+abc_len+bc_minusa)
    solo_c=len(c)-(bc_minusa+abc_len+ac_minusb)
    
    return solo_a,solo_b,ab_minusc,solo_c,ac_minusb,bc_minusa,abc_len

################################## Investigating misclassified features/probs ##########################################  
def full_sub_idx_creator(b,sub_b,test_lens):
    
    sub_b_idx=np.array([np.where(b==sub_b[i])[0][0] for i in range(len(sub_b))],dtype='int')
    
    full_idxs=full_idx_creator(test_lens)
    
    return  [full_idxs[sub_b_idx[j]] for j in range(len(sub_b_idx))]

def sub_idx_creator(b,sub_b):
    
    sub_b_idx=np.array([np.where(b==sub_b[i])[0][0] for i in range(len(sub_b))],dtype='int')

    return sub_b_idx

def full_correct_wrong_idx_features(correct_ids,ids,lens,probs=None,X=None,y=None):
    
    false_ids=np.array([ids[i] for i in range(len(ids)) if ids[i] not in correct_ids],dtype='int')
    
    full_correct_idxs=full_sub_idx_creator(ids,correct_ids,lens)
    full_wrong_idxs=full_sub_idx_creator(ids,false_ids,lens)
    
    if probs is not None:
        
        assert (X is not None) & (y is not None)
        
        correct_probs=[probs[full_correct_idxs[i]] for i in range(len(full_correct_idxs))]
        wrong_probs=[probs[full_wrong_idxs[i]] for i in range(len(full_wrong_idxs))]
        
        correct_features=[X[full_correct_idxs[i]] for i in range(len(full_correct_idxs))]
        wrong_features=[X[full_wrong_idxs[i]] for i in range(len(full_wrong_idxs))]
        
        y_wrong=[np.array(y,dtype='int')[full_wrong_idxs[i]][0] for i in range(len(full_wrong_idxs))]
        
        return full_correct_idxs,full_wrong_idxs,correct_probs,wrong_probs,correct_features,wrong_features,y_wrong
    else:
        return full_correct_idxs,full_wrong_idxs

###################################  introduce new metrics for counting how many severity cases ##################################
def cutoff(data,threds=[5,10]):
    
    num_kind=len(threds)
    
    if num_kind==len(data):
        output=np.array([len(np.where(data[i]>threds[i])[0]) for i in range(num_kind)])
        if num_kind>2:
            output[2]=len(np.where(data[2]<threds[2])[0])     
        
    else:
        
        output=np.array([len(np.where(data[:,i]>threds[i])[0]) for i in range(num_kind)])
        if num_kind>2:
            output[2]=len(np.where(data[:,2]<threds[2])[0])    
            
    return output

def cutoff_proportion(data,threds=[5,10]):

    num_kind=len(threds)
    lens=len(data[0])
    
    if num_kind==len(data):
        output=np.array([len(np.where(data[i]>threds[i])[0])/lens for i in range(num_kind)])
        if num_kind>2:
            output[2]=len(np.where(data[2]<threds[2])[0])/lens
    else:
        output=np.array([len(np.where(data[:,i]>threds[i])[0])/lens for i in range(num_kind)])        
        if num_kind>2:
            output[2]=len(np.where(data[:,2]<threds[2])[0])/lens
    

    return output


def mean_severity_feature(minlen=20,data_type='weekly',job='cla',proportion=True,threds=[5,10]):
    
    mean_severity=[]
    pseudo_sliding_mean_severity=[]
    y_=[]
    job_int=int(job=='reg')
    
    X_Original=load_pickle(DATA_interim+'participants_class_'+data_type+'.pkl')

    for j in range(len(X_Original)):
    
        len_=len(X_Original[j].data[0])
        
        if len_>=minlen+job_int:
            if proportion:
                mean_severity_=cutoff_proportion(X_Original[j].data,threds=threds)
            else:
                mean_severity_=cutoff(X_Original[j].data)
                
            mean_severity.append(mean_severity_)
            
            y_.append(X_Original[j].diagnosis)
            
            for start in np.arange(len_-minlen-job_int+1):
                pseudo_sliding_mean_severity.append(mean_severity_) 
                
    mean_severity=np.array(mean_severity)
    pseudo_sliding_mean_severity=np.array(pseudo_sliding_mean_severity)
    
    return mean_severity,y_, pseudo_sliding_mean_severity

def sliding_mean_severity_feature(X_original,proportion=True):
    
    if proportion:
        sliding_mean_severity=[cutoff_proportion(X_original[j]) for j in range(len(X_original))]
    else:
        sliding_mean_severity=[cutoff(X_original[j]) for j in range(len(X_original))]
        
    return np.asarray(sliding_mean_severity)
################################## For spectrums on triangles ##########################################  
def prob_individual(probs):
    score=np.zeros(probs.shape[-1])
    preds=np.argmax(probs,axis=1)
    for i in range(len(score)):
        score[i]+=len(np.where(preds==i)[0])
    return score/len(preds)


def trianglePoints_generator(lens,probs,y,mental_dict={0:"borderline",1:"healthy",2:"bipolar"}):
    
    full_idxs=full_idx_creator(lens)
    preds=np.array([prob_individual(probs[full_idxs[i]]) for i in range(len(lens))])
    y_labels=np.array([y[full_idxs[i][0]] for i in range(len(lens))],dtype='int')


    

    trianglePoints={ "borderline":  [],
                     "healthy":     [],
                     "bipolar":     []}
    
    for j in range(len(y_labels)):
    
        trianglePoints[mental_dict[y_labels[j]]].append(preds[j])

    return trianglePoints


def plotDensityMap(scores,title=None):
    """Plots, given a set of scores, the density map on a triangle.

    Parameters
    ----------
    scores : list
        List of scores, where each score is a 3-dimensional list.

    """


    TRIANGLE = np.array([[math.cos(math.pi*0.5), math.sin(math.pi*0.5)],
                        [math.cos(math.pi*1.166), math.sin(math.pi*1.166)],
                        [math.cos(math.pi*1.833), math.sin(math.pi*1.833)]])

    
#     scores1=[]
#     for score in scores:
#         for element in score:
#             scores1.append(element)
        
        
    pointsX = [score.dot(TRIANGLE)[0] for score in scores]
    pointsY = [score.dot(TRIANGLE)[1] for score in scores]

    vertices = []
    vertices.append(np.array([1,0,0]).dot(TRIANGLE))
    vertices.append(np.array([0,1,0]).dot(TRIANGLE))
    vertices.append(np.array([0,0,1]).dot(TRIANGLE))
    for i in range(3):
        p1 = vertices[i]
        if i == 2:
            p2 = vertices[0]
        else:
            p2 = vertices[i+1]
        c = 0.5 * (p1 + p2)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', linestyle='-', linewidth=2)
        plt.plot([0, c[0]], [0, c[1]], color='k', linestyle='-', linewidth=1)



    ax = plt.gca()
    ax.set_xlim([-1.2, 1.32])
    ax.set_ylim([-0.7,1.3])

    ax.text(0.8, -0.6,"bipolar")
    ax.text(-1.1, -0.6, "healthy" )
    ax.text(-0.15, 1.05, "borderline")


    data = [[pointsX[i], pointsY[i]] for i in range(len(pointsX))]

    H, _, _=np.histogram2d(pointsX,pointsY,bins=40,normed=True)
    norm=H.max()-H.min()
  
    contour1=0.75
    target1=norm*contour1+H.min()
    def objective(limit, target):
        w = np.where(H>limit)
        count = H[w]
        return count.sum() - target


  #  level1 = scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
 #   levels = [level1]
    
    sns.kdeplot(np.array(pointsX), np.array(pointsY),shade=True, ax=ax)
    sns.kdeplot(np.array(pointsX), np.array(pointsY), n_levels=3, ax=ax, cmap="Reds")
 #   sns.kdeplot(np.array(pointsX), np.array(pointsY),shade=True, ax=ax,cbar=True,cmap=plt.cm.RdYlGn)
    if title is not None:
        plt.savefig(title+".jpeg",dpi=300)
    else:
        plt.show()


def trim_triangle(col,index=1):
    
    """trim healthy data such that plot can be seen.

    Parameters
    ----------
    col : a collection of healthy data

    Returns
    -------
    list of str
        List of data can has been trim by threshold 0.03.

    """
   

    try1=copy.deepcopy(col)
    
    other_index=[i for i in range(3) if i!=index]
    
    for md in try1:
        if md[int(index)]>0.4:
            
            md[int(index)]+=(md[other_index[0]]+md[other_index[-1]])/2
            md[other_index[0]]*=0.5
            md[other_index[-1]]*=0.5
 
    return try1
def trim_triangle_HC(col):
    
    """trim healthy data such that plot can be seen.

    Parameters
    ----------
    col : a collection of healthy data

    Returns
    -------
    list of str
        List of data can has been trim by threshold 0.03.

    """
   

    try1=copy.deepcopy(col)
    HC_index=1
    other_index=[i for i in range(3) if i!=HC_index]
    
    for md in try1:
        
        if md[int(HC_index)]>0.5:
            
            md[int(HC_index)]+=(md[other_index[0]]+md[other_index[-1]])*3/5
            md[other_index[0]]*=2/5
            md[other_index[-1]]*=2/5  
            
    for index in other_index:
        t=0
        other_index_=[i for i in range(3) if i!=index]
    
        for md in try1:
            
            if index<other_index[-1]:
                
                if md[int(index)]>0.2:
                    t+=1
                    if t%2==1:
                        md[int(index)]+=(md[other_index_[0]]+md[other_index_[-1]])*15/16
                        md[other_index_[0]]*=1/16
                        md[other_index_[-1]]*=1/16
    
                    if t%2==0:
                        md[int(index)]+=(md[other_index_[0]]+md[other_index_[-1]])*2/4
                        md[other_index_[0]]*=2/4
                        md[other_index_[-1]]*=2/4
            else:           
                if md[int(index)]>0.55:
                    t+=1
                    if t%2==1:
                        md[int(index)]+=(md[other_index_[0]]+md[other_index_[-1]])*3/4
                        md[other_index_[0]]*=1/4
                        md[other_index_[-1]]*=1/4
    
                    if t%2==0:
                        md[int(index)]+=(md[other_index_[0]]+md[other_index_[-1]])*2/4
                        md[other_index_[0]]*=2/4
                        md[other_index_[-1]]*=2/4        
    return try1

#########################################################################################
def metric_features_many_per_par(data_cat='weekly',minlen=10,subsubfolder='_RM_cleaned/'):
    

    data_subdir=DATA_processed+data_cat+'/cla/'+str(minlen)+'/many_per_par'+subsubfolder
    
    name='X_instability_metrics.npy'
    X_metrics=np.load(data_subdir+name)
    
    name_='X_pvars.npy'
    X_pvars=np.load(data_subdir+name_)
    
    X_TKEOs=X_metrics[:,:X_metrics.shape[-1]//3]
    X_RMSSDs=X_metrics[:,X_metrics.shape[-1]//3:X_metrics.shape[-1]*2//3]
    X_SSSDs=X_metrics[:,X_metrics.shape[-1]*2//3:]
    return X_TKEOs,X_RMSSDs,X_SSSDs,X_metrics,X_pvars