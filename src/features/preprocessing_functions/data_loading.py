import h5py
import os

from definitions import *
from src.omni.functions import *



def loadParticipants(sub_path,save=True):
    """Loads the participant cohort.
    
    Parameters
    ----------
    path : str
        Location of the directory with the data.

    Returns
    -------
    list of Participant for ALTMAN test, time list, id list, QS list
    
    list of Participant for QIDS test, time list, id list, QS list
    
     list of Participant for both ALTMAN & QIDS test, time list, id list, QS list

    """

    
    participants_list=[]
    participants_dataALTMAN=[]
    participants_timeALTMAN=[]

    
    participants_dataQIDS=[]
    participants_timeQIDS=[]
    
    path=DATA_raw+sub_path
    
    for filename in sorted(os.listdir(path)):
        f = h5py.File(path+filename, 'r')
        
        k = list(f.keys())
        
        if "QIDS" in f['data'] and "ALTMAN" in f['data']:
            participants_list.append(filename.split("-")[0])
            participants_dataALTMAN.append(f[k[1]]['ALTMAN']['data'][()][0])
            participants_dataQIDS.append(f[k[1]]['QIDS']['data'][()][0])

            participants_timeALTMAN.append(f[k[1]]['ALTMAN']['time'][()][0])
            participants_timeQIDS.append(f[k[1]]['QIDS']['time'][()][0])   
          
            

            
    participant_data_list=[]
    participant_data_list.append(participants_dataALTMAN)
    participant_data_list.append(participants_dataQIDS)

    
    participants_time_list=[]
    participants_time_list.append(participants_timeALTMAN)
    participants_time_list.append(participants_timeQIDS)
    
    if save:
        save_pickle(participants_list,DATA_interim+'participants_list.pkl')
        save_pickle(participant_data_list,DATA_interim+'participants_data.pkl')        
        save_pickle(participants_time_list,DATA_interim+'participants_time.pkl')
        
#     return participants_list,participant_data_list,participants_time_list

def loadParticipants_general(sub_path,save=True):
    """Loads the participant cohort.
    
    Parameters
    ----------
    path : str
        Location of the directory with the data.

    Returns
    -------
    list of Participant for ALTMAN test, time list, id list, QS list
    
    list of Participant for QIDS test, time list, id list, QS list
    
     list of Participant for both ALTMAN & QIDS test, time list, id list, QS list

    """

    
    participants_list=[]
    participants_dataALTMAN=[]
    participants_timeALTMAN=[]

    
    participants_dataQIDS=[]
    participants_timeQIDS=[]

    participants_dataEQD5=[]
    participants_timeEQD5=[]
    
    participants_dataGAD=[]
    participants_timeGAD=[]
    
    path=DATA_raw+sub_path
    
    for filename in sorted(os.listdir(path)):
        f = h5py.File(path+filename, 'r')
        
        k = list(f.keys())
        
        if "QIDS" in f['data'] and "ALTMAN" in f['data'] and "EQ_5D" in f['data'] and "GAD_7" in f['data']:
            participants_list.append(filename.split("-")[0])
            participants_dataALTMAN.append(f[k[1]]['ALTMAN']['data'][()][0])
            participants_dataQIDS.append(f[k[1]]['QIDS']['data'][()][0])
            participants_dataEQD5.append(f[k[1]]['EQ_5D']['data'][()][0])
            participants_dataGAD.append(f[k[1]]['GAD_7']['data'][()][0])
            
            participants_timeALTMAN.append(f[k[1]]['ALTMAN']['time'][()][0])
            participants_timeQIDS.append(f[k[1]]['QIDS']['time'][()][0])   
            participants_timeEQD5.append(f[k[1]]['EQ_5D']['time'][()][0])
            participants_timeGAD.append(f[k[1]]['GAD_7']['time'][()][0])          
            

            
    participant_data_list=[]
    participant_data_list.append(participants_dataALTMAN)
    participant_data_list.append(participants_dataQIDS)
    participant_data_list.append(participants_dataEQD5)
    participant_data_list.append(participants_dataGAD)
    
    participants_time_list=[]
    participants_time_list.append(participants_timeALTMAN)
    participants_time_list.append(participants_timeQIDS)
    participants_time_list.append(participants_timeEQD5)
    participants_time_list.append(participants_timeGAD)
    
    if save:
        save_pickle(participants_list,DATA_interim+'participants_list_general.pkl')
        save_pickle(participant_data_list,DATA_interim+'participants_data_general.pkl')        
        save_pickle(participants_time_list,DATA_interim+'participants_time_general.pkl')
        
#     return participants_list,participant_data_list,participants_time_list