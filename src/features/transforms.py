import numpy as np
import iisignature
import copy

from rpy2.robjects.packages import importr

pvar=importr('pvar')

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def standardise_sym(x,l1,l2):
    """standardise data x to [-1,1]


    Parameters
    
    x: original data    
    l1: lower bound
    l2: upper bound
    ----------

    Returns
    -------
    
    standardised data


    """    
    
    return (x-l1)*2/(l2-l1)-1


def standardise(x,l1,l2):
    """standardise data x to [0,1]


    Parameters
    
    x: original data    
    l1: lower bound
    l2: upper bound
    ----------

    Returns
    -------
    
    standardised data


    """  
   
    return (x-l1)/float(l2-l1)

def counting_processing(data,missing_value=-1.,count_time=False):
    """
    time=False: where data having time as one coordinate
    
    """
    new_data=np.zeros((data.shape[0],data.shape[1]+1))
    new_data[:,:-1]=data
    
    num_kind=data.shape[1]-int(count_time)
    
    all_miss_row_ids=np.concatenate([np.where(new_data[:,i]==missing_value)[0] for i in range(num_kind)])
    unique,counts=np.unique(all_miss_row_ids,return_counts=True)
    
    new_data[:,-1][unique]+=counts
    
    new_data[:,-1]=np.cumsum(new_data[:,-1])
    return new_data
    
def ffill(data,missing_value=-1.0,start_replace=0):
    """
    """
        
    data_new=np.zeros((data.shape[0],data.shape[1]))
    
    data_new[:,:]=data[:,:]
    
    for j in range(data_new.shape[1]):
        missing_ids=np.where(data_new[:,j]==missing_value)[0]
        missing_num=len(np.where(data_new[:,j]==missing_value)[0])
        
        if missing_num>0:
            
            if missing_num<data_new.shape[0]:
                start_replace=(np.sum(data_new)+missing_num)/(data_new.shape[0]-missing_num)
            
            if data_new[0,j]==missing_value:
                    data_new[0,j]=start_replace

            for idx in missing_ids:
                data_new[idx,j]=data_new[idx-1,j]

    return data_new

    
    
# def normalise(data,scoreMAX=[20,27],scoreMIN=[0,0],cumsum=True):
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
#     normalised_data[:,-1]=data[:,-1]
    
#     len_data=len(scoreMAX)


#     for i in range(len_data):
        
#         normalised_data[:,i]=standardise_sym(data[:,i],scoreMIN[i],scoreMAX[i])
    
#     if data.shape[1]-len_data>1:
#         for ii in range(data.shape[1]-len_data)[:-1]:
        
#             idx=len_data+ii
#             if data[-1,idx]!=0:
#                 normalised_data[:,idx]=standardise(data[:,idx],0,data[-1,idx])

                
#     if cumsum:
#             normalised_data=np.cumsum(normalised_data,axis=0)
    
#     return normalised_data

def normalise(data,scoreMAX=[20,27],scoreMIN=[0,0],time_normalise=True,cumsum=True):
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
    normalised_data[:,len(scoreMAX):]=data[:,len(scoreMAX):]
    
    len_data=len(scoreMAX)


    for i in range(len_data):
        
        normalised_data[:,i]=standardise_sym(data[:,i],scoreMIN[i],scoreMAX[i])
    
    if time_normalise:
        if data.shape[1]-len_data>1:
            for ii in range(data.shape[1]-len_data)[:-1]:  
                idx=len_data+ii
                if data[-1,idx]!=0:
                    normalised_data[:,idx]=standardise(data[:,idx],0,data[-1,idx])
        
            time_data=normalised_data[:,idx]
        
    if cumsum:
#             if data.shape[1]-len_data>1:
#                 normalised_data=np.cumsum(normalised_data,axis=0)  
#                 normalised_data[:,idx]=time_data
#             else:
                normalised_data=np.cumsum(normalised_data,axis=0)
#                 normalised_data[:,:len(scoreMAX)]=np.cumsum(normalised_data[:,:len(scoreMAX)],axis=0)
    
    return normalised_data


###########################################################################################


def clinical_mean_without_missing(data,missing_value=-1.0):

    """clean missing data with strategy 1.


    Parameters
    ----------
     data:  orignal data in list form.



    Returns
    -------
    newdata: data without missing one in array form.

    """
    
    means=np.empty((0,1),float)
    
    for j in range(data.shape[1]):
        
        data1=data[:,j]
        
        missing_ids=np.where(data1==missing_value)[0]
        missing_num=len(np.where(data1==missing_value)[0])
        
        if missing_num>0:
            
            if missing_num<data.shape[0]:
                means=np.append(means,(np.sum(data1)+missing_num)/(data.shape[0]-missing_num))
            else:
                means=np.append(means,-1)
        else:
                means=np.append(means,np.mean(data1))
                
    return means
################################### metrics ################################################
def TKEO(data):
    
        """ 
    
        TKEO metric for sequential data
    
        """
    
        data=np.array(data)
        output=0
    
        if len(data)>=2:

            for j in range(len(data))[1:-1]:
            
                output+=(data[j]**2-data[j+1]*data[j-1])/len(data)
            
            return output

def RMSSD(data):
    
    """ 
    
    RMSSD metric for sequential data
    
    """
    data=np.array(data)    
    
    output=0
    
    if len(data)>=2:
        
        return np.sqrt(np.sum((data[1:]-data[:-1])**2)/len(data))

#     if output==0:
#         return output+10**(-5)
#     else:
#         return np.sqrt(output)

def customised_SSSD(data,weight=5):
    
    """ 
    
    RMSSD metric for sequential data
    
    """
    data=np.array(data)    
    
    if len(data)>=2:

        return np.sum(((data[1:]-data[:-1])/weight)**2)/len(data)

def instability_metric_vec(data,metric='TKEO',weights=None):
        
        kind_num=data.shape[-1]
        
        output=np.zeros(kind_num)
        
        if len(data[:,0])>=2:
            
            if metric=='TKEO':
            
                return np.array([TKEO(data[:,i]) for i in range(kind_num)])
            
            elif metric=='RMSSD':
                return np.array([RMSSD(data[:,i]) for i in range(kind_num)])
            
            elif metric=='SSSD':
                
                return np.array([customised_SSSD(data[:,i],weight=weights[i]) for i in range(kind_num)])
            
            else:
                return np.concatenate([np.array([TKEO(data[:,i]) for i in range(kind_num)]),\
                                       np.array([RMSSD(data[:,i]) for i in range(kind_num)]),\
                                       np.array([customised_SSSD(data[:,i],weight=weights[i]) for i in range(kind_num)])])


def pvar_(data,order=3):

    """ 
    
    p-variation for sequential data, need to call R package pvar
    
    """
    
    data=np.array(data)        
    output=0
    
    if len(data)>=2:
        output+=pvar.pvar(data,order)[0][0]
        
    return output

def pvar_vec(data,order=3):
        
        kind_num=data.shape[-1]
        
        output=np.zeros(kind_num)
        
        if len(data[:,0])>=2:
            
            return np.array([pvar_(data[:,i]) for i in range(kind_num)])
########################################### Signature Related ###############################


def lead_lag(path,time=True):
    """
        Lead_lag transform 
    
    """
    (num,dim)=path.shape

    times=np.arange(num)


    lead_path=np.zeros((num*3-3,dim))
    lag_path=np.zeros((num*3-3,dim))
    expand_time=np.zeros(num*3-3)
    
    for i in range(num)[:-1]:
        lead_path[3*i,:]=path[i+1,:]
        lead_path[3*i+1,:]=path[i+1,:]
        lead_path[3*i+2,:]=path[i+1,:]

        lag_path[3*i,:]=path[i,:]
        lag_path[3*i+1,:]=path[i,:]
        lag_path[3*i+2,:]=path[i+1,:]


        expand_time[3*i]=2*i*times[i]
        expand_time[3*i+1]=(2*i+1)*times[i]
        expand_time[3*i+2]=(2*i+3.0/2)*times[i]


    lead_path=np.array(lead_path)
    lag_path=np.array(lag_path)
    
    new_path1=np.concatenate((lead_path, lag_path),axis=1)
    if time:
        return new_path1
    else:
        new_path=np.concatenate((new_path1,np.array([expand_time]).T/num),axis=1)
        return new_path

def rectilinear_interpolation(data1,data2):
    
    indices=np.where(data1-data2!=0.0)[0]
    
    
    if len(indices)>1:
            
        data_inter=np.zeros((len(indices), len(data1)))
        
        for i in range(len(indices))[:-1]:
        
            index_now=indices[i]
            if index_now!=0 and index_now!=len(data2)-1:
                data_inter[i,:]=data_inter[i-1,:]
                data_inter[i,index_now]=data2[index_now]

            
            elif index_now==0:
            
                data_inter[i,0]=data2[0]
                data_inter[i,1:]=data1[1:]
                
                
        data_inter[-1,:]=data2
    
        
    
    else:
        data_inter=np.zeros((1, len(data1)))
        
        data_inter=data2.reshape(-1,1).T
        
        
    return data_inter    
    

def rectilinear_output(data):
    
    data1=copy.deepcopy(data)
    
    data1=data1[:1,:]

    for i in range(data.shape[0])[1:]:

        insert=rectilinear_interpolation(data[i-1,:],data[i,:])

        data1=np.append(data1,insert,axis=0)

    return data1

def pen_on_pen_off(data, initial=True):

 

    """

 

    pen_on_pen_off function is the visibility transformation on discrete data

 
    Input:

 

    data: (n,d)-shape numpy format data,

           n: number of observations; d: dimension

 

    initial: True or False

             True: initial position is important

             False: tail position is important

 

    Output:

 

    

    pened_data: (n+2,d+1)-shape numpy format data

    

 

    """    

 

    pened_data=np.zeros((data.shape[0]+2,data.shape[1]+1))

    if initial:

 
        pened_data[2:,:-1]=data
        pened_data[1,:-1]=data[0,:]
        pened_data[2:,-1]=np.ones(data.shape[0])

    else:

        pened_data[:-2,:-1]=data
        pened_data[-2,:-1]=data[-1,:]
        pened_data[:-2,-1]=np.ones(data.shape[0])        

    return pened_data



def signature_transform(x,M,operator=lambda x: x):
    """
        signature transform for x after operator
    """
    return iisignature.sig(operator(x), M)


def log_signature_transform(x,M=3,s=iisignature.prepare(4 ,3),operator=lambda x: x):
    """
        signature transform for x after operator
    """
    path=operator(x)
    return iisignature.logsig(path,s)
