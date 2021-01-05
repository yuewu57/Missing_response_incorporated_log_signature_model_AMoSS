import sys
import pickle
import os

sys.path.insert(0, '../../')

from definitions import *

from src.features.original_feature_extraction import *
from src.omni.functions import load_pickle
from src.features.main_feature_functions import *

if __name__ == '__main__':

    minlen=20
    save=True
    
    scoreMAX,scoreMIN,cumsum=[20,27,100,21],[0,0,0,0],True
    
    weights=[3,3,10,3]
    
    subdir_name='/many_per_par_general/'
    
    for data_cat in data_cats: 
        
        print("Current data category:",data_cat)
        data=load_pickle(DATA_interim+"participants_class_"+data_cat+"_general.pkl")

        for regression in [False, True][:1]:
            
            subdir=data_cat+'/'+name_reg[int(regression)]+'/'+str(minlen)+subdir_name
            print('Saving to ',subdir)
#             instability_metrics_many_per_par(data,minlen=minlen, metric='',weights=weights,subdir=subdir,save=save) 
#             for time in [False, True]:
                
#                 for count in [False, True]:
                    
#                     for normalise in [False, True]:
                
#                         print("regression or not:",regression,"; time-augmented or not:",time,\
#                               "; count or not:",count,"; normalise or not:",normalise)
                
              
#                         building_data_many_per_par(data,minlen=minlen,regression=regression, count=count,time=time, \
#                                                    scoreMAX=scoreMAX,scoreMIN=scoreMIN,cumsum=cumsum,\
#                                                    normalise=normalise,subdir=subdir,save=save)
        
#         print('\n')
        
    start_replace=0
    for data_cat in data_cats: 
        
            print("Current data category:",data_cat)

            for time in [False, True]:              
                for count in [False, True]:
                    for normalise in [False, True]:
                
                    
                        print("regression or not:",regression,"; time-augmented or not:",time,\
                              "; count or not:",count,"; normalise or not:",normalise)
                
                        subdir=data_cat+'/'+name_reg[int(regression)]+'/'+str(minlen)+subdir_name
                        data_subdir=DATA_processed+subdir

                        if not time and not count:
                            X=load_pickle(data_subdir+"X.pkl") 
                            name='_'
                        elif time and not count:
                            if normalise:
                                X=load_pickle(data_subdir+"X_normalised_time_auged.pkl")
                                name='_normalised_time_auged_'
                            else:
                                X=load_pickle(data_subdir+"X_time_auged.pkl")
                                name='_time_auged_'
                        elif time and count:
                            if normalise:
                                X=load_pickle(data_subdir+"X_normalised_count_time_auged.pkl")
                                name='_normalised_count_time_auged_'
                            else:
                                X=load_pickle(data_subdir+"X_count_time_auged.pkl")
                                name='_count_time_auged_'
                        elif count and not time:
                            if normalise:
                                X=load_pickle(data_subdir+"X_normalised_count.pkl")
                                name='_normalised_count_'
                            else:
                                X=load_pickle(data_subdir+"X_count.pkl")
                                name='_count_'
                        
#                         for order in [2,3]:
#                             print("for order:", order)
#                             for leadlag in [False,True][:1]:
#                                 for log in [False, True][1:]:
#                                     for rectilinear in [False, True][:1]:
#                                         Sig_features=signature_featuring(X,order, time=time,start_replace=start_replace,\
#                                                                      leadlag=leadlag,rectilinear=rectilinear,log=log)
                        
#                                         names=name+str(int(leadlag))+'_'+str(int(rectilinear))
#                                         if log:
#                                             sig_name='logsig_'+str(order)
#                                         else:
#                                             sig_name='sig_'+str(order)
                                    
#                                         np.save(data_subdir+sig_name+"_features"+names+".npy",Sig_features)
                        
                       
#                         print('Processing ffill features ',name)
#                         ffill_features=ffill_featuring(X,start_replace=start_replace)
#                         np.save(data_subdir+"ffill_mean_features"+name+".npy",ffill_features)
                
#                         print('Processing data-mean features without missing vaues')
#                         mean_features=mean_featuring(X)
#                         np.save(data_subdir+"mean_features"+name+".npy",mean_features)

                        print('Processing flatten features:')
                        flatten_features=flatten_featuring(X)
                        np.save(data_subdir+"flatten_features"+name+".npy",flatten_features)

                
                        print('\n')
                
      
        
        