import os
import random
import numpy as np
import copy
import matplotlib.dates as mdates
from datetime import date

    
from functools import reduce
    
from definitions import *

def multiple_to_one(data,rule="average"):
    """
    input:
    data, 1d array
    rule: can be 'average'/'max'/'last'
    
    output:
    
        if data contains all -1, then -1
        else, from valid values, using the rule to ouput scalar
    
    """

    valid_idxs=np.where(data!=-1)[0]

    output=int(-1)
    
    if len(valid_idxs)!=0:
        if rule=='average':
            output=np.mean(data[valid_idxs])
        elif rule=='max':
            output=np.max(data[valid_idxs])
        elif rule=='first':
            output=data[valid_idxs[0]]
        else:
            output=data[valid_idxs[-1]]
    else:
        output=-1

    return output


def multiple_to_one_list(data_list,rule='average'):
    
    """
    Input:
    data_list: [[data 1],[data2 ],[data3],....]
    
    Output: 1d numpy array
            [rule on data 1, rule on data2,rule on data3,....]
    """
    
    
    return np.array([multiple_to_one(data_list[i],rule=rule) for i in range(len(data_list))])

#     data_lens=np.array([len(data_list[i]) for i in range(len(data_list))])

#     idx_multiple_data=np.where(data_lens>1)[0]
    
#     for idx in idx_multiple_data:
#         data_list[idx]=np.array([multiple_to_one(data_list[idx],rule=rule)])     
      
#     return np.array(data_list).reshape(-1,1)

def time_data_splitting(time_data,target_datas):
    """
    
    
    Input:
    time_data: 1d array
    target_datas;[[data_seq1],[data_seq2],...]

    
    Output:
    
    unique_times:1d array, unique values in time_data
    
    datas_per_turn: [[data_seq1 per turn],[data_seq2 per turn],..]
    
                where [data_seq1 per turn]=[[data_subseq for first unique time],
                                             [data_subseq for the second unique time],..]
                                             
                eg, time_data=[1,1,2]
                    data_seq1=[6,7,8]
                    data_seq2=[2,5,5]
                    then
                    unique_times=[1,2]
                    [data_seq1 per turn]=[[6,7],[8]]
                    [data_seq2 per turn]=[[2,5],[5]]
                    so datas_per_turn: [[[6,7],[8]],=[[2,5],[5]]]
    """
    
    unique_times, unique_times_indices = np.unique(time_data, return_index=True)


    datas_per_turn=[[] for i in range(len(target_datas))]
    
    extended_unique_times_indices=np.append(unique_times_indices,len(time_data))
    lengths=extended_unique_times_indices[1:]-unique_times_indices

        

    for i in range(len(extended_unique_times_indices))[:-1]:
        start_idx=extended_unique_times_indices[i]
        end_idx=extended_unique_times_indices[i+1]
        
        for i in range(len(target_datas)):
            target_data=target_datas[i]
            data_this_turn=target_data[start_idx:end_idx]
            datas_per_turn[i].append(data_this_turn)
    
    return  unique_times, datas_per_turn

def cleaning_same_day_data(Participants,rule="average"):
    
    """cleaning redundant data: 
    
        if two data are stored in the same day, and scores are the same, keep either;
        if one score is recorded as missing, then keep the other one.


    Parameters
    
    class participant data
    ----------

    Returns
    -------
    
    shortened participant data


    """    
    
    Pars=copy.deepcopy(Participants)
    
    n=len(Pars[0].data)
    
    for par in Pars:
        for i in range(n):
            
            unique_times,[datas_per_turn]=time_data_splitting(par.time[i],[par.data[i]])
            
            par.time[i]=unique_times
            par.data[i]=multiple_to_one_list(datas_per_turn,rule=rule)
                
    #sanity check
    for par in Pars:
        for i in range(n):

            total_len=len(par.time[i])  
            for j in range(total_len)[:-1]:
                assert int(par.time[i][j+1])!=int(par.time[i][j])

    return Pars    

##################################################################
def filling(shorter_time_list,longer_time_list,data_list,keep='short'):
    
    """
    
        if short:
            longer_time_list is time for data_list
            -------> keep data_list with shorter time_list
        
        else:
             shorter_time_list is time for data_list
             -------> expanded data_list with longer time_list
        
    """
    
    if keep=='short':
        output=np.zeros_like(shorter_time_list)-1.0
        
    else:
        output=np.zeros_like(longer_time_list)-1.0
        
    output_idxs=np.array([i for i in range(len(longer_time_list))\
                              if longer_time_list[i] in shorter_time_list],dtype='int')

    if keep=='short':        
        output+=data_list[output_idxs]+1
    else:
        output[output_idxs]+=data_list+1
        
    return output
    
def data_expanded(list1,list2,time1,time2,rule='union'):
    """
        if union:
            we take union of time1 and time2 and expand list1 and list 2 by filling -1
        else:
            we take intersection of time1 and time2 and shrink list1 and list 2 accordingly
    
    """
    output1,output2=list1,list2
    expanded_time=time1
    
    if len(time1)!=len(time2) or len(np.where(time1-time2!=0)[0])>0:

        if rule=='union':
            expanded_time_set=set(time1) | set(time2)
        else:
            expanded_time_set=set(time1).intersection(time2)
        
        expanded_time=np.array(sorted(list(expanded_time_set)))

        if rule=='union':
            output1=filling(time1,expanded_time,list1,keep='long')
            output2=filling(time2,expanded_time,list2,keep='long')
        else:
            output1=filling(expanded_time,time1,list1,keep='short')
            output2=filling(expanded_time,time2,list2,keep='short')
        
        assert len(output1)==len(output2)
        
    return output1,output2,expanded_time,expanded_time

def aligning_data(Participants,rule='union'):
    """
    To fix the problem when len(altman) is not equal to len(qids)
    
    """
    Par=copy.deepcopy(Participants)
    
    Par_save=[]
    
    for par in Par: 
        
        par.data[0],par.data[1],par.time[0],par.time[1]=data_expanded(par.data[0],\
                                                                      par.data[1],\
                                                                      par.time[0],\
                                                                      par.time[1],\
                                                                      rule=rule)
        if par.idNumber not in exclude_ids:
            
            Par_save.append(par)
            

    save_pickle(Par_save,DATA_interim+"participants_class_aligned.pkl")
      
    return Par
    
def aligning_data_general(Participants,rule='union'):
    """
    To fix the problem when len(altman) is not equal to len(qids)
    
    """
    Par=copy.deepcopy(Participants)
    
    num_kind=len(Par[0].data)

    Par_save=[]
    
    for par in Par: 
        
        for i in range(num_kind):
            idx1=i
            if idx1+1>=num_kind:
                idx2=(int(idx1+1) % num_kind)
            else:
                
                idx2=i+1
            
            
            
            par.data[idx1],par.data[idx2],par.time[idx1],par.time[idx2]=data_expanded(par.data[idx1],\
                                                                                      par.data[idx2],\
                                                                                      par.time[idx1],\
                                                                                      par.time[idx2],\
                                                                                      rule=rule)
            
        par.data[0],par.data[1],par.time[0],par.time[1]=data_expanded(par.data[0],\
                                                                                      par.data[1],\
                                                                                      par.time[0],\
                                                                                      par.time[1],\
                                                                                      rule=rule)  
        
        if par.idNumber not in exclude_ids:
            
            Par_save.append(par)
            

    save_pickle(Par_save,DATA_interim+"participants_class_aligned_general.pkl")
      
    return Par




#############################################################################################
def dates_difference(s1,s2):
    
    d1=date(mdates.num2date(s1).year,mdates.num2date(s1).month,mdates.num2date(s1).day)
                                          
    d2=date(mdates.num2date(s2).year,mdates.num2date(s2).month,mdates.num2date(s2).day)
    
    delta = d2 - d1
                                          
    return delta.days

def weeks_splitting(time1,asrm,qids):
    """
    Input:
    time1: 1d numpy array, machine time since 1970
    asrm,qids: 1d numpy array
    
    Output:
    
    unique_weeks_per_year: [[unique weeks for year1],..[unique weeks for year n],...]
    times_per_week_per_year: [[times per week for year 1],...,[times per week for year n],..]
                              where [times per week for year 1]=[[times for week1, year1],...,[times for week n, year 1],...] 
     asrm_per_week_per_year: regroup asrm in format like times_per_week_per_year
     qids_per_week_per_year: similar as above
    """
    length=len(time1)
    years=np.array([mdates.num2date(time1[i]).isocalendar()[0] for i in range(length)])
    weeks=np.array([mdates.num2date(time1[i]).isocalendar()[1] for i in range(length)])


    
    unique_years,[weeks_per_year,times_per_year,\
                 asrm_per_year,qids_per_year]= time_data_splitting(years,\
                                                                   [weeks,time1,\
                                                                    asrm,qids])    

    unique_weeks_per_year=[]

    times_per_week_per_year=[]
    asrm_per_week_per_year=[]
    qids_per_week_per_year=[]
    
    for i in range(len(weeks_per_year)):
        
        unique_weeks_this_year,\
        [times_per_week_this_year, asrm_per_week_this_year,\
         qids_per_week_this_year]= time_data_splitting(weeks_per_year[i],\
                                                       [times_per_year[i],\
                                                        asrm_per_year[i],\
                                                        qids_per_year[i]])
        
        unique_weeks_per_year.append(unique_weeks_this_year)
        times_per_week_per_year.append(times_per_week_this_year)
        asrm_per_week_per_year.append(asrm_per_week_this_year)
        qids_per_week_per_year.append(qids_per_week_this_year)

    return unique_weeks_per_year,\
            times_per_week_per_year,\
            asrm_per_week_per_year,\
            qids_per_week_per_year
            
def weeks_splitting_general(time1,data_list):
    """
    Input:
    time1: 1d numpy array, machine time since 1970
    data_list: list of several 1d numpy arrays
    
    Output:
    
    unique_weeks_per_year: [[unique weeks for year1],..[unique weeks for year n],...]
    times_per_week_per_year: [[times per week for year 1],...,[times per week for year n],..]
                              where [times per week for year 1]=[[times for week1, year1],...,[times for week n, year 1],...] 
     asrm_per_week_per_year: regroup asrm in format like times_per_week_per_year
     qids_per_week_per_year: similar as above
    """
    length=len(time1)
    years=np.array([mdates.num2date(time1[i]).isocalendar()[0] for i in range(length)])
    weeks=np.array([mdates.num2date(time1[i]).isocalendar()[1] for i in range(length)])


    
    unique_years,new_timed_data_list= time_data_splitting(years, [weeks,time1]+data_list)  

    weeks_per_year,times_per_year=new_timed_data_list[0],new_timed_data_list[1]
    data_list_per_year=new_timed_data_list[2:] 
    
    unique_weeks_per_year=[]

    times_per_week_per_year=[]
    data_list_per_week_per_year=[[] for j in range(len(data_list))]

    
    for i in range(len(weeks_per_year)):
        
        current_set=[times_per_year[i]]

        for j in range(len(data_list)):
            current_set.append(data_list_per_year[j][i])
            
        unique_weeks_this_year,\
        current_set_per_week_this_year= time_data_splitting(weeks_per_year[i],current_set)
        
        times_per_week_this_year=current_set_per_week_this_year[0]
        unique_weeks_per_year.append(unique_weeks_this_year)
        times_per_week_per_year.append(times_per_week_this_year)
        
        for j in range(len(data_list)):
            data_list_per_week_per_year[j].append(current_set_per_week_this_year[j+1])

    return unique_weeks_per_year,\
            times_per_week_per_year,\
            data_list_per_week_per_year


def calendar_week_cleaning_per_par(unique_weeks_per_year,\
                                   times_per_week_per_year,\
                                   asrm_per_week_per_year,\
                                   qids_per_week_per_year,\
                                   rule='first'):
    """
        grouping times/asrm/qids in each calendar week, and handling multiple entries within the same week by given rule
        
    Input: output from weeks_splitting
    
    Output: cleaned asrm/qids/asrm time/qids time
        
    """
    num_years=len(unique_weeks_per_year)
    times,asrms,qids=[],[],[]
    
    for i in range(num_years):
        unique_weeks_this_year,times_per_week_this_year,\
            asrm_per_week_this_year,qids_per_week_this_year=unique_weeks_per_year[i],\
                                                            times_per_week_per_year[i],\
                                                            asrm_per_week_per_year[i],\
                                                            qids_per_week_per_year[i]
      
        times_per_week_this_year=multiple_to_one_list(times_per_week_this_year,rule=rule).astype('int')      
        asrm_per_week_this_year=multiple_to_one_list(asrm_per_week_this_year,rule=rule)
        qids_per_week_this_year=multiple_to_one_list(qids_per_week_this_year,rule=rule)
        

        if len(unique_weeks_this_year)>1:
            gaps=unique_weeks_this_year[1:]-unique_weeks_this_year[:-1]
            gaps_where=np.where(gaps>1)[0]
            
            if len(gaps_where)>=1:
                gap_lens=gaps[gaps_where]-1
                
                missing_fills=[np.array([-1 for k in range(gap_len)]) for gap_len in gap_lens]
                times_fills=[(np.arange(gap_lens[j])+1)*7+times_per_week_this_year[gaps_where[j]]\
                            for j in range(len(gap_lens))]
                week_fills=[np.arange(gap_lens[j])+1+unique_weeks_this_year[:-1][gaps_where[j]]\
                            for j in range(len(gap_lens))]
                
                idx_fills=gaps_where+1
                
                for kk in range(len(gaps_where)):
                    
                    kk_=len(gaps_where)-1-kk
                    idx_fill=idx_fills[kk_]
                    unique_weeks_this_year=np.insert(unique_weeks_this_year,\
                                                      idx_fill,\
                                                     week_fills[kk_])
                    
                    times_per_week_this_year=np.insert(times_per_week_this_year,\
                                                      idx_fill,\
                                                      times_fills[kk_])

                    asrm_per_week_this_year=np.insert(asrm_per_week_this_year,\
                                                       idx_fill,\
                                                      missing_fills[kk_])
                    qids_per_week_this_year=np.insert(qids_per_week_this_year,\
                                                     idx_fill,\
                                                     missing_fills[kk_])
        times.append(times_per_week_this_year)
        asrms.append(asrm_per_week_this_year)
        qids.append(qids_per_week_this_year)
        
#         print(asrm_per_week_this_year)
        assert len(np.where(unique_weeks_this_year[1:]-unique_weeks_this_year[:-1]>1)[0])==0
        
                
    return np.concatenate(asrms),np.concatenate(qids),\
            np.concatenate(times),np.concatenate(times)        

def calendar_week_cleaning_per_par_general(unique_weeks_per_year,\
                                           times_per_week_per_year,\
                                           data_list_per_week_per_year,\
                                           rule='first'):
    """
        grouping times/asrm/qids in each calendar week, and handling multiple entries within the same week by given rule
        
    Input: output from weeks_splitting
    
    Output: cleaned asrm/qids/asrm time/qids time
        
    """
    num_years=len(unique_weeks_per_year)

    num_kinds=len(data_list_per_week_per_year)

    times=[]
    data_list=[[] for m in range(num_kinds)]
    
    for i in range(num_years):
        unique_weeks_this_year,times_per_week_this_year=unique_weeks_per_year[i],\
                                                        times_per_week_per_year[i]
        
        data_list_per_week_this_year=[[] for m in range(num_kinds)]
        for m in range(num_kinds):
            data_list_per_week_this_year[m].append(data_list_per_week_per_year[m][i])

        times_per_week_this_year=multiple_to_one_list(times_per_week_this_year,rule=rule).astype('int')  
        
        for m in range(num_kinds):        
            data_list_per_week_this_year[m]=multiple_to_one_list(data_list_per_week_this_year[m][0],rule=rule)


        if len(unique_weeks_this_year)>1:
            gaps=unique_weeks_this_year[1:]-unique_weeks_this_year[:-1]
            gaps_where=np.where(gaps>1)[0]
            
            if len(gaps_where)>=1:
                gap_lens=gaps[gaps_where]-1
                
                missing_fills=[np.array([-1 for k in range(gap_len)]) for gap_len in gap_lens]
                times_fills=[(np.arange(gap_lens[j])+1)*7+times_per_week_this_year[gaps_where[j]]\
                            for j in range(len(gap_lens))]
                week_fills=[np.arange(gap_lens[j])+1+unique_weeks_this_year[:-1][gaps_where[j]]\
                            for j in range(len(gap_lens))]
                
                idx_fills=gaps_where+1
                
                for kk in range(len(gaps_where)):
                    
                    kk_=len(gaps_where)-1-kk
                    idx_fill=idx_fills[kk_]
                    unique_weeks_this_year=np.insert(unique_weeks_this_year,\
                                                      idx_fill,\
                                                     week_fills[kk_])
                    
                    times_per_week_this_year=np.insert(times_per_week_this_year,\
                                                      idx_fill,\
                                                      times_fills[kk_])
                    
                    for m in range(num_kinds):
                        data_list_per_week_this_year[m]=np.insert(data_list_per_week_this_year[m],\
                                                                     idx_fill,missing_fills[kk_])
                    

        times.append(times_per_week_this_year)
        for m in range(num_kinds):
            data_list[m].append(data_list_per_week_this_year[m])

        assert len(np.where(unique_weeks_this_year[1:]-unique_weeks_this_year[:-1]>1)[0])==0
    
    for m in range(num_kinds):

        data_list[m]=np.concatenate(data_list[m])
                
    return  data_list,np.concatenate(times),np.concatenate(times)        


      

    
def cleaning_sameweek_data(Participants,rule='first',save=True):
    
    Pars=copy.deepcopy(Participants)

    for par in Pars:

        unique_weeks_per_year,\
        times_per_week_per_year,\
        asrm_per_week_per_year,\
        qids_per_week_per_year=weeks_splitting(par.time[0],par.data[0],par.data[1])
        
        par.data[0],par.data[1],\
        par.time[0],par.time[1]=calendar_week_cleaning_per_par(unique_weeks_per_year,\
                                                               times_per_week_per_year,\
                                                               asrm_per_week_per_year,\
                                                               qids_per_week_per_year,\
                                                               rule=rule)
    
    if save:
        save_pickle(Pars,DATA_interim+"participants_class_weekly.pkl")
    else:  
        return Pars
    
def cleaning_sameweek_data_general(Participants,rule='first',save=True):
    
    Pars=copy.deepcopy(Participants)
    
    for par in Pars:

        unique_weeks_per_year, times_per_week_per_year,\
                    data_list_per_week_per_year=weeks_splitting_general(par.time[0],par.data)
        
        data_list,par.time[0],par.time[1]=calendar_week_cleaning_per_par_general(unique_weeks_per_year,\
                                                                                 times_per_week_per_year,\
                                                                                 data_list_per_week_per_year,\
                                                                                 rule=rule)
        par.data=data_list
        
    num_kind=len(par.data)
    if save:
        save_pickle(Pars,DATA_interim+"participants_class_weekly_general.pkl")
    else:  
        return Pars
    
# def clearning_redundant_missing_data(subdir_to_participants,save=True):

#     participants=load_pickle(DATA_interim+subdir_to_participants)
#     pars=copy.deepcopy(participants)
    
#     for par in pars:
        
#         idxs_missings=[]
        
#         for i in range(len(par.data)):
    
#             idxs_missings.append(np.where(par.data[i]==-1)[0])

#             idxs_missings_intersect=reduce(np.intersect1d, tuple(idxs_missings))    

#             if len(par.data[0])-1 in idxs_missings_intersect:
                
#                 some_set=np.where(idxs_missings_intersect[1:]-idxs_missings_intersect[:-1]!=1)[0]
#                 if len(some_set)!=0:
#                     idxs_missing_start=idxs_missings_intersect[some_set[-1]+1]
#                 else:
#                     idxs_missing_start=idxs_missings_intersect[0]
#                 for i in range(len(par.data)):
#                     par.data[i]=par.data[i][:idxs_missing_start]
                
#     if save:
#         save_pickle(pars,DATA_interim+subdir_to_participants.split('.')[0]+'_RM_cleaned.pkl')
#     else:  
#         return pars
