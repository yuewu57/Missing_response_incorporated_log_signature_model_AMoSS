import numpy as np


from src.features.transforms import *


def calling_operators(leadlag=False,time=True,rectilinear=False,vt=True):
    if not vt:
        if leadlag and rectilinear:       
            return lambda x: rectilinear_output(lead_lag(x,time=time))  
        elif leadlag and not rectilinear:
            return lambda x: lead_lag(x,time=time)
        elif not leadlag and rectilinear:
            return lambda x: rectilinear_output(x)
        else:
            return lambda x: x
    else:
        if leadlag and rectilinear:       
            return lambda x: pen_on_pen_off(rectilinear_output(lead_lag(x,time=time)))
        elif leadlag and not rectilinear:
            return lambda x: pen_on_pen_off(lead_lag(x,time=time))
        elif not leadlag and rectilinear:
            return lambda x: pen_on_pen_off(rectilinear_output(x))
        else:
            return lambda x: pen_on_pen_off(x)       
    
def signature_featuring(inputs,order,time=False,start_replace=0, leadlag=False,\
                        rectilinear=False, log=False,vt=False):


    
    """process data using signatures before fitting into machine learning models.

    Parameters
    ----------
    inputs : list
        The numpy data.

    order : int, optional
        Order of the signature.
        Default is 2.
        
    time: True/False, including time or not
    
    count: True/False, using missing count or not
    
    start_replace: the number to fill in place if the first element is missing when conducting forward filling
    
    leadlag:True/False, using leadlag transform or not
    
    rectilinear:True/False, using rectilinear transform or not
    
    Returns
    -------
    
    x,y in appropriate form
    
    """


    X=[]
    t=0
    
    operator=calling_operators(leadlag=leadlag,time=(~time),rectilinear=rectilinear,vt=vt)

    if log:
        for i in range(len(inputs)):
            
            x=inputs[i]

            if t==0:
                dim=operator(x).shape[-1]
                s=iisignature.prepare(dim,order)
                
            sig=log_signature_transform(x,order,s=s,operator=operator)
            X.append(sig)
            t+=1
    else:
        
        for i in range(len(inputs)):
            
            x=inputs[i]

            sig=signature_transform(x,order,operator=operator)
 
            X.append(sig)
    
    return np.asarray(X)

def signature_featuring_before_cumsum(inputs,order, time=True,count=True,start_replace=0, leadlag=False,\
                                      rectilinear=False,log=False,count_time=False,time_normalise=True,\
                                      scoreMAX=[20,27],scoreMIN=[0,0],vt=False,cumsum=True):


    
    """process data using signatures before fitting into machine learning models.

    Parameters
    ----------
    inputs : list
        The numpy data.

    order : int, optional
        Order of the signature.
        Default is 2.
        
    time: True/False, including time or not
    
    count: True/False, using missing count or not
    
    start_replace: the number to fill in place if the first element is missing when conducting forward filling
    
    leadlag:True/False, using leadlag transform or not
    
    rectilinear:True/False, using rectilinear transform or not
    
    Returns
    -------
    
    x,y in appropriate form
    
    """


    X=[]
    t=0
    
    operator=calling_operators(leadlag=leadlag,time=(~time),rectilinear=rectilinear,vt=vt)

    if log:
        for i in range(len(inputs)):
            
            x=inputs[i]

            if count:
                x=counting_processing(x,count_time=count_time)

            
            x_=normalise(ffill(x,start_replace=start_replace),\
                         scoreMAX=scoreMAX,scoreMIN=scoreMIN,\
                         time_normalise=time_normalise,cumsum=cumsum)
            
            if t==0:
                dim=operator(x_).shape[-1]
                s=iisignature.prepare(dim,order)
                
            sig=log_signature_transform(x_,order,s=s,operator=operator)
            X.append(sig)
            t+=1
    else:
        
        for i in range(len(inputs)):
            
            x=inputs[i]

            if count:
                x=counting_processing(x,time=time)
            
            x_=normalise(ffill(x,start_replace=start_replace),scoreMAX=scoreMAX,scoreMIN=scoreMIN,cumsum=cumsum)
        
            sig=signature_transform(x_,order,operator=operator)
 
            X.append(sig)
    
    return np.asarray(X)

def ffill_featuring(inputs,start_replace=0):


    
    """process data using forward filling only before fitting into machine learning models.

    Parameters
    ----------
    inputs : list
        The numpy data.

    Returns
    -------
    
    x in appropriate form
    
    """
        

    return np.asarray([np.mean(ffill(inputs[i],start_replace=start_replace),axis=0) for i in range(len(inputs))])

def mean_featuring(inputs):

    
    """process data using means before fitting into machine learning models.

    Parameters
    ----------
    input : list of numpy data 

    Returns
    -------
    
    x in appropriate form
    
    """

        

    return np.asarray([clinical_mean_without_missing(inputs[i]) for i in range(len(inputs))])


def flatten_featuring(inputs):

    
    """process data using means before fitting into machine learning models.

    Parameters
    ----------
    input : list of numpy data 

    Returns
    -------
    
    x in appropriate form
    
    """

    inputs=np.array(inputs)


    return np.asarray([inputs[i].flatten() for i in range(len(inputs))])