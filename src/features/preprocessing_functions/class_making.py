import os
import csv
import random
import numpy as np


from definitions import *

Save_dir=DATA_interim
            
            
def make_classes(save=True):
    
    """data process to make class Participant

    Parameters
    
    List of corresponding test1 & test2, test1 time & test 2 time, id list
    ----------

    Returns
    -------
    class of participants for the corresponding 2 tests


    """
    participants_list=load_pickle(DATA_interim+'participants_list.pkl')
    participants_data_list=load_pickle(DATA_interim+'participants_data.pkl')
    participants_time_list=load_pickle(DATA_interim+'participants_time.pkl')
    
    num=len(participants_list)
    
    Participants=sorted(list(csv.reader(open(DATA_raw+"patients-Copy1.csv"))))
    
    participants=[]
    
    t=0
    
    for i in range(num):

        n=int(participants_list[i])
        
        for l in Participants:
            if int(l[0])==n:
                
                if not l[1].isdigit():

                    print(n)
                    
                    break
                    

                data=[participants_data_list[j][i].astype(int) for j in range(len(participants_data_list)) ]

                time=[participants_time_list[j][i].astype(int) for j in range(len(participants_time_list)) ]
                
                                
                                
                bp0 = int(l[1])
                
                bp = {1: 2, 2: 0, 3: 1}[bp0]
                
                participant=Participant(data, time, n, bp,None)
                participants.append(participant)
                t+=1
                break

    if save:
        save_pickle(participants,DATA_interim+"participants_class.pkl")
    else:   
        return participants

def make_classes_general(save=True):
    
    """data process to make class Participant

    Parameters
    
    List of corresponding test1 & test2, test1 time & test 2 time, id list
    ----------

    Returns
    -------
    class of participants for the corresponding 2 tests


    """
    participants_list=load_pickle(DATA_interim+'participants_list_general.pkl')
    participants_data_list=load_pickle(DATA_interim+'participants_data_general.pkl')
    participants_time_list=load_pickle(DATA_interim+'participants_time_general.pkl')
    
    num=len(participants_list)
    
    Participants=sorted(list(csv.reader(open(DATA_raw+"patients-Copy1.csv"))))
    
    participants=[]
    
    t=0
    
    for i in range(num):

        n=int(participants_list[i])
        
        for l in Participants:
            if int(l[0])==n:
                
                if not l[1].isdigit():

                    print(n)
                    
                    break
                    

                data=[participants_data_list[j][i].astype(int) for j in range(len(participants_data_list)) ]

                time=[participants_time_list[j][i].astype(int) for j in range(len(participants_time_list)) ]
                
                                
                                
                bp0 = int(l[1])
                
                bp = {1: 2, 2: 0, 3: 1}[bp0]
                
                participant=Participant(data, time, n, bp,None)
                participants.append(participant)
                t+=1
                break

    if save:
        save_pickle(participants,DATA_interim+"participants_class_general.pkl")
    else:   
        return participants


