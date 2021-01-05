import os
import csv
import random
import numpy as np
import pickle
import copy

from definitions import *
from src.omni.functions import save_pickle,_create_folder_if_not_exist
from src.features.transforms import standardise_sym, standardise, ffill,instability_metric_vec,pvar_vec

def excluding_participant(collection,minlen=10):
    """
        Excluding participants who do not have enough data
    """
    collection1=copy.deepcopy(collection)
    
    total_len=len(collection1)
    
    len_perpaticipant=np.array([len(collection1[i].data[0]) for i in range(total_len)])
    idx_to_exclude=np.where(len_perpaticipant<minlen)[0]

    
    for i in range(len(idx_to_exclude)):
        idx_now=idx_to_exclude[len(idx_to_exclude)-1-i]

        collection1.remove(collection1[idx_now])
    
    return collection1

def building_data_one_per_par(collection,minlen=10,regression=False,time=False,subdir=None,save=False):

    """Builds the training and out-of-sample sets.


    Parameters
    ----------
     collection :  data in class format
     


    Returns
    -------
    list
        x: list of Numpy data set, columns: different features; rows: different time observations
        time: relative time is added for each participant (eg, initial time for participant is a,
                                                        then the relative time is current time -a )
    list
        y: ground truth
        If regression, pair of next day [asrm score,qids score] 

    """
    
    random.seed(42)
    reg_binary=int(regression)
    
    collection1=excluding_participant(collection,minlen=minlen+reg_binary)

    total_len=len(collection1)
        
    num_kind=len(collection1[0].data)

    x=[]
    y=[]
    for j in range(total_len):
        
        current_participant=collection1[j]
        seq_len=len(current_participant.data[0])

        assert seq_len>=minlen+reg_binary
            
        if seq_len>minlen+reg_binary:
            
            random_start=random.randint(0,seq_len-minlen-reg_binary)
            current_data=np.array([current_participant.data[i][random_start:random_start+minlen] for i in range(num_kind)])
            if time:
                current_time=(current_participant.time[0][random_start:random_start+minlen]-current_participant.time[0][0])/7
                current_data=np.concatenate([current_data,current_time.reshape(-1,1).T])
            
            x.append(current_data.T)
            if regression:
                current_y=np.array([int(current_participant.data[i][random_start+minlen]) for i in range(num_kind)])

                y.append(current_y)
            else:
                y.append(current_participant.diagnosis)
                
        else:
            current_data=np.array([current_participant.data[i][0:minlen] for i in range(num_kind)])
            if time:
                current_time=(current_participant.time[0][0:minlen]-current_participant.time[0][0])/7
                current_data=np.concatenate([current_data,current_time.reshape(-1,1).T])

                
            if regression:
                current_y=np.array([int(current_participant.data[i][minlen]) for i in range(num_kind)])
                
            x.append(current_data.T)
            y.append(current_participant.diagnosis)
    
    if save:        
        if subdir==None:
            subdir=DATA_processed
        else:
            subdir=DATA_processed+subdir
            _create_folder_if_not_exist(subdir)
        
        
        if time:
            save_pickle(x,subdir+"X_time_auged.pkl")
            save_pickle(y,subdir+"Y_time_auged.pkl")
        else:
            save_pickle(x,subdir+"X.pkl")            
            save_pickle(y,subdir+"Y.pkl")
            
    else:
        return x,y


def building_data_many_per_par_before_cumsum(collection,minlen=10,regression=False,time=False,subdir=None,save=False):

    """Builds the training and out-of-sample sets.


    Parameters
    ----------
     collection :  data in class format
     


    Returns
    -------
    list
        x: list of Numpy data set, columns: different features; rows: different time observations
        time: relative time is added for each participant (eg, initial time for participant is a,
                                                        then the relative time is current time -a )
    list
        y: ground truth
        If regression, pair of next day [asrm score,qids score] 

    """
    
    random.seed(42)
    reg_binary=int(regression)
    
    collection1=excluding_participant(collection,minlen=minlen+reg_binary)

    total_len=len(collection1)
        
    num_kind=len(collection1[0].data)

    x=[]
    y=[]
    lens=[]
    ids=[]
    
    for j in range(total_len):
        
        current_participant=collection1[j]
        ids.append(current_participant.idNumber)
        
        seq_len=len(current_participant.data[0])
        
        assert seq_len>=minlen+reg_binary

            
        for start in np.arange(seq_len-minlen-reg_binary+1):
                
                current_data=np.array([current_participant.data[i][start:start+minlen] for i in range(num_kind)])
                if time:
                    current_time=(current_participant.time[0][start:start+minlen]-current_participant.time[0][0])/7
                    ###newly added
                    current_time_diff=current_time[1:]-current_time[:-1]
                    current_time[1:]=current_time_diff
                    #########
                    current_data=np.concatenate([current_data,current_time.reshape(-1,1).T])
            
                x.append(current_data.T)
                if regression:
                    current_y=np.array([int(current_participant.data[i][start+minlen]) for i in range(num_kind)])

                    y.append(current_y)
                else:
                    y.append(current_participant.diagnosis)
        
        lens.append(seq_len-minlen-reg_binary+1)
    
    if save:        
        if subdir==None:
            subdir=DATA_processed+'before_cumsum/'
        else:
            subdir=DATA_processed+subdir
            _create_folder_if_not_exist(subdir)
        
        
        if time:
            save_pickle(x,subdir+"X_time_auged_before_cumsum.pkl")
        else:
            save_pickle(x,subdir+"X_before_cumsum.pkl") 
            
        save_pickle(y,subdir+"Y.pkl")
            
        np.save(subdir+"lens.npy",np.array(lens)) 
        save_pickle(ids,subdir+"ids.pkl")
    else:
        return x,y

def counting(data,missing_value=-1.):
    """
    time=False: where data having time as one coordinate
    
    """
    counts=np.zeros(len(data[0]))

    num_kind=len(data)


    all_miss_row_ids=np.concatenate([np.where(data[i]==missing_value)[0] for i in range(num_kind)])
    unique,count=np.unique(all_miss_row_ids,return_counts=True)

    counts[unique]+=count
    
    counts=np.cumsum(counts)

    return counts

       

# def normalising(data,scoreMAX=[20,27],scoreMIN=[0,0],cumsum=True):
#     """Normalises the data of the patient with missing count.

#     Parameters
#     ----------
#     data : numpy data, [number of observations, number of features]
#     scoreMAX: max scores for asrm and qids
#     scoreMIN: min scores for asrm and qids 
    
#     Returns
#     -------
#     normalised_data: data that are normalised and cumulated if cumsum=True.

#     """

#     normalised_data=np.zeros((data.shape[0],data.shape[1]))

#     len_data=len(scoreMAX)

#     for i in range(len_data):        
#         normalised_data[:,i]=standardise_sym(data[:,i],scoreMIN[i],scoreMAX[i])
        
#     if cumsum:
#         normalised_data[:,:len(scoreMAX)]=np.cumsum(normalised_data[:,:len(scoreMAX)],axis=0)
    
#     return normalised_data

def normalising_general(data,scoreMAX=[20,27,100,21],scoreMIN=[0,0,0,0],cumsum=True):
    """Normalises the data of the patient with missing count.

    Parameters
    ----------
    data : numpy data, [number of observations, number of features]
    scoreMAX: max scores for asrm and qids
    scoreMIN: min scores for asrm and qids 
    
    Returns
    -------
    normalised_data: data that are normalised and cumulated if cumsum=True.

    """

    normalised_data=np.zeros((data.shape[0],data.shape[1]))

    len_data=len(scoreMAX)

    for i in range(len_data):        
        normalised_data[:,i]=standardise_sym(data[:,i],scoreMIN[i],scoreMAX[i])
        
    if cumsum:
        normalised_data[:,:len(scoreMAX)]=np.cumsum(normalised_data[:,:len(scoreMAX)],axis=0)
    
    return normalised_data

def building_data_many_per_par(collection,minlen=10,regression=False,time=False,count=True,\
                               scoreMAX=[20,27,100,21],scoreMIN=[0,0,0,0],cumsum=True,\
                               normalise=False, subdir=None,save=False):

    """Builds the training and out-of-sample sets.


    Parameters
    ----------
     collection :  data in class format
     


    Returns
    -------
    list
        x: list of Numpy data set, columns: different features; rows: different time observations
        time: relative time is added for each participant (eg, initial time for participant is a,
                                                        then the relative time is current time -a )
    list
        y: ground truth
        If regression, pair of next day [asrm score,qids score] 

    """
    
    random.seed(42)
    reg_binary=int(regression)
    
    collection1=excluding_participant(collection,minlen=minlen+reg_binary)

    total_len=len(collection1)
        
    num_kind=len(collection1[0].data)

    x=[]
    y=[]
    lens=[]
    ids=[]
    
    for j in range(total_len):
        
        current_participant=collection1[j]
        ids.append(current_participant.idNumber)
        
        seq_len=len(current_participant.data[0])
        
         
        assert seq_len>=minlen+reg_binary
        
        if count:
            current_count_=counting(current_participant.data)
            
            if current_count_[-1]!=0 and normalise:
                current_count_=standardise(current_count_,0,current_count_[-1])
                
        data=np.array([current_participant.data[i] for i in range(num_kind)]).T
        
        if normalise:
            data=normalising_general(ffill(data),scoreMAX=scoreMAX,scoreMIN=scoreMIN,cumsum=cumsum)
        
        if time:
            current_time_=(current_participant.time[0]-current_participant.time[0][0])/7
            
            if current_time_[-1]!=0 and normalise:
                current_time_=standardise(current_time_,0,current_time_[-1])
            
        for start in np.arange(seq_len-minlen-reg_binary+1):
                
                current_data=data[start:start+minlen,:]
                
                if time:
                    
                    current_time=current_time_[start:start+minlen]
                    current_data=np.concatenate([current_data,current_time.reshape(-1,1)],axis=1)
                    
                    
                if count:
                    
                    current_count=current_count_[start:start+minlen]
                    current_data=np.concatenate([current_data,current_count.reshape(-1,1)],axis=1)
                    
                x.append(current_data)
                
                if regression:
                    current_y=np.array([int(current_participant.data[i][start+minlen]) for i in range(num_kind)])

                    y.append(current_y)
                else:
                    y.append(current_participant.diagnosis)
        
        lens.append(seq_len-minlen-reg_binary+1)

    
    if save:        
        if subdir==None:
            subdir=DATA_processed
        else:
            subdir=DATA_processed+subdir
            _create_folder_if_not_exist(subdir)
        
        if not time and not count:
            if normalise:
                save_pickle(x,subdir+"X.pkl") 
            else:
                save_pickle(x,subdir+"raw_X.pkl") 
        elif time and not count:
            if normalise:
                save_pickle(x,subdir+"X_normalised_time_auged.pkl")
            else:
                save_pickle(x,subdir+"X_time_auged.pkl")
        elif time and count:
            if normalise:
                save_pickle(x,subdir+"X_normalised_count_time_auged.pkl")
            else:
                save_pickle(x,subdir+"X_count_time_auged.pkl")
        elif count and not time:
            if normalise:
                save_pickle(x,subdir+"X_normalised_count.pkl")
            else:

                save_pickle(x,subdir+"X_count.pkl")

            
        save_pickle(y,subdir+"Y.pkl")
            
        np.save(subdir+"lens.npy",np.array(lens)) 
        save_pickle(ids,subdir+"ids.pkl")
    else:
        return x,y

def building_data_many_per_par_noGAD7(collection,minlen=10,regression=False,time=False,count=True,\
                               scoreMAX=[20,27,100],scoreMIN=[0,0,0],cumsum=True,\
                               normalise=False, subdir=None,save=False):

    """Builds the training and out-of-sample sets.


    Parameters
    ----------
     collection :  data in class format
     


    Returns
    -------
    list
        x: list of Numpy data set, columns: different features; rows: different time observations
        time: relative time is added for each participant (eg, initial time for participant is a,
                                                        then the relative time is current time -a )
    list
        y: ground truth
        If regression, pair of next day [asrm score,qids score] 

    """
    
    random.seed(42)
    reg_binary=int(regression)
    
    collection1=excluding_participant(collection,minlen=minlen+reg_binary)

    total_len=len(collection1)
        
    num_kind=len(collection1[0].data)-1

    x=[]
    y=[]
    lens=[]
    ids=[]
    
    for j in range(total_len):
        
        current_participant=collection1[j]
        ids.append(current_participant.idNumber)
        
        seq_len=len(current_participant.data[0])
        
        par_data=[current_participant.data[i] for i in range(num_kind)]
         
        assert seq_len>=minlen+reg_binary
        
        if count:
            current_count_=counting(par_data)
            
            if current_count_[-1]!=0 and normalise:
                current_count_=standardise(current_count_,0,current_count_[-1])
                
        data=np.array([par_data[i] for i in range(num_kind)]).T
        
        if normalise:
            data=normalising_general(ffill(data),scoreMAX=scoreMAX,scoreMIN=scoreMIN,cumsum=cumsum)
        
        if time:
            current_time_=(current_participant.time[0]-current_participant.time[0][0])/7
            
            if current_time_[-1]!=0 and normalise:
                current_time_=standardise(current_time_,0,current_time_[-1])
            
        for start in np.arange(seq_len-minlen-reg_binary+1):
                
                current_data=data[start:start+minlen,:]
                
                if time:
                    
                    current_time=current_time_[start:start+minlen]
                    current_data=np.concatenate([current_data,current_time.reshape(-1,1)],axis=1)
                    
                    
                if count:
                    
                    current_count=current_count_[start:start+minlen]
                    current_data=np.concatenate([current_data,current_count.reshape(-1,1)],axis=1)
                    
                x.append(current_data)
                
                if regression:
                    current_y=np.array([int(current_participant.data[i][start+minlen]) for i in range(num_kind)])

                    y.append(current_y)
                else:
                    y.append(current_participant.diagnosis)
        
        lens.append(seq_len-minlen-reg_binary+1)

    
    if save:        
        if subdir==None:
            subdir=DATA_processed
        else:
            subdir=DATA_processed+subdir
            _create_folder_if_not_exist(subdir)
        
        if not time and not count:
            if normalise:
                save_pickle(x,subdir+"X.pkl") 
            else:
                save_pickle(x,subdir+"raw_X.pkl") 
        elif time and not count:
            if normalise:
                save_pickle(x,subdir+"X_normalised_time_auged.pkl")
            else:
                save_pickle(x,subdir+"X_time_auged.pkl")
        elif time and count:
            if normalise:
                save_pickle(x,subdir+"X_normalised_count_time_auged.pkl")
            else:
                save_pickle(x,subdir+"X_count_time_auged.pkl")
        elif count and not time:
            if normalise:
                save_pickle(x,subdir+"X_normalised_count.pkl")
            else:

                save_pickle(x,subdir+"X_count.pkl")

            
        save_pickle(y,subdir+"Y.pkl")
            
        np.save(subdir+"lens.npy",np.array(lens)) 
        save_pickle(ids,subdir+"ids.pkl")
    else:
        return x,y
    
def instability_metrics_many_per_par(collection,minlen=10, metric='TKEO',weights=None,order=3,subdir=None,save=False):

    """Builds the training and out-of-sample sets.


    Parameters
    ----------
     collection :  data in class format
     


    Returns
    -------
    list
        x: list of Numpy data set, columns: different features; rows: different time observations
        time: relative time is added for each participant (eg, initial time for participant is a,
                                                        then the relative time is current time -a )
    list
        y: ground truth
        If regression, pair of next day [asrm score,qids score] 

    """
    
    random.seed(42)

    
    collection1=excluding_participant(collection,minlen=minlen)

    total_len=len(collection1)
        
    num_kind=len(collection1[0].data)

    x=[]
    x_pvars=[]
    
    for j in range(total_len):
        
        current_participant=collection1[j]
        
        seq_len=len(current_participant.data[0])
                 
        assert seq_len>=minlen
                        
        data=np.array([current_participant.data[i] for i in range(num_kind)]).T
        data=ffill(data)
                    
        for start in np.arange(seq_len-minlen+1):
                
                current_data=data[start:start+minlen,:]
                                
                x.append(instability_metric_vec(current_data,metric=metric,weights=weights))
                        
                x_pvars.append(pvar_vec(current_data,order=order))
    
    if save:        
        if subdir==None:
            subdir=DATA_processed
        else:
            subdir=DATA_processed+subdir
            _create_folder_if_not_exist(subdir)
        
        
        np.save(subdir+"X_instability_metrics.npy",np.array(x)) 
        np.save(subdir+"X_pvars.npy",np.array(x_pvars)) 
        
    else:
        return x,x_pvars
            
            

            
# def building_data_many_per_par(collection,minlen=10,regression=False,time=False,count=True,subdir=None,save=False):

#     """Builds the training and out-of-sample sets.


#     Parameters
#     ----------
#      collection :  data in class format
     


#     Returns
#     -------
#     list
#         x: list of Numpy data set, columns: different features; rows: different time observations
#         time: relative time is added for each participant (eg, initial time for participant is a,
#                                                         then the relative time is current time -a )
#     list
#         y: ground truth
#         If regression, pair of next day [asrm score,qids score] 

#     """
    
#     random.seed(42)
#     reg_binary=int(regression)
    
#     collection1=excluding_participant(collection,minlen=minlen+reg_binary)

#     total_len=len(collection1)
        
#     num_kind=len(collection1[0].data)

#     x=[]
#     y=[]
#     lens=[]
#     ids=[]
    
#     for j in range(total_len):
        
#         current_participant=collection1[j]
#         ids.append(current_participant.idNumber)
        
#         seq_len=len(current_participant.data[0])
        
         
#         assert seq_len>=minlen+reg_binary
        
#         if count:
#             current_count_=counting(current_participant.data)
        
#         data=np.array([current_participant.data[i][start:start+minlen] for i in range(num_kind)])
        
#         ffill(data,missing_value=-1.0,start_replace=0)
# #         if seq_len>minlen+reg_binary:
            
#         for start in np.arange(seq_len-minlen-reg_binary+1):
                
#                 current_data=np.array([current_participant.data[i][start:start+minlen] for i in range(num_kind)])
#                 if time:
#                     current_time=(current_participant.time[0][start:start+minlen]-current_participant.time[0][0])/7
#                     ###newly added
# #                     current_time_diff=current_time[1:]-current_time[:-1]
# #                     current_time[1:]=current_time_diff
#                     #########
#                     current_data=np.concatenate([current_data,current_time.reshape(-1,1).T])
                
#                 if count:
                    
#                     current_count=current_count_[start:start+minlen]
#                     current_data=np.concatenate([current_data,current_count.reshape(-1,1).T])
                    
#                 x.append(current_data.T)
#                 if regression:
#                     current_y=np.array([int(current_participant.data[i][start+minlen]) for i in range(num_kind)])

#                     y.append(current_y)
#                 else:
#                     y.append(current_participant.diagnosis)
        
#         lens.append(seq_len-minlen-reg_binary+1)
# #         else:
# #             current_data=np.array([current_participant.data[i][0:minlen] for i in range(num_kind)])
# #             if time:
# #                 current_time=current_participant.time[0][0:minlen]-current_participant.time[0][0]
# #                 current_data=np.concatenate([current_data,current_time.reshape(-1,1).T])

                
# #             if regression:
# #                 current_y=np.array([int(current_participant.data[i][minlen]) for i in range(num_kind)])
                
# #             x.append(current_data.T)
# #             y.append(current_participant.diagnosis)
    
#     if save:        
#         if subdir==None:
#             subdir=DATA_processed
#         else:
#             subdir=DATA_processed+subdir
#             _create_folder_if_not_exist(subdir)
        

#         if time and not count:
#             save_pickle(x,subdir+"X_time_auged.pkl")
#         elif time and count:
#             save_pickle(x,subdir+"X_count_time_auged.pkl")
#         elif count and not time:
#             save_pickle(x,subdir+"X_count.pkl")
#         else:
#             save_pickle(x,subdir+"X.pkl")            
#         save_pickle(y,subdir+"Y.pkl")
            
#         np.save(subdir+"lens.npy",np.array(lens)) 
#         save_pickle(ids,subdir+"ids.pkl")
#     else:
#         return x,y
