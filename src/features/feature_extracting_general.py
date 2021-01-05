import sys
import pickle
import os

sys.path.insert(0, '../../')

from definitions import *

from src.features.original_feature_extraction import *
from src.omni.functions import load_pickle
from src.features.main_feature_functions import *

if __name__ == '__main__':

    minlen=10
    save=True
    
    scoreMAX,scoreMIN,cumsum=[20,27,100,21],[0,0,0,0],True
    
    subdir_name='/many_per_par_general/'
    
    regression,time,count,normalise=False, True,True,False
    
    for data_cat in data_cats: 
        
        print("Current data category:",data_cat)
        data=load_pickle(DATA_interim+"participants_class_"+data_cat+"_general.pkl")
    
        subdir=data_cat+'/'+name_reg[int(regression)]+'/'+str(minlen)+subdir_name
        print('Saving to ',subdir)
        print("regression or not:",regression,"; time-augmented or not:",time, "; count or not:",count,"; normalise or not:",normalise)
                
              
        building_data_many_per_par(data,minlen=minlen,regression=regression, count=count,time=time, \
                                                   scoreMAX=scoreMAX,scoreMIN=scoreMIN,cumsum=cumsum,\
                                                   normalise=normalise,subdir=subdir,save=save)
        
        print('\n')
        
    start_replace,order,leadlag,rectilinear,log=0,3,False,False,True
    for data_cat in data_cats: 
        
            print("Current data category:",data_cat)
                  
            print("regression or not:",regression,"; time-augmented or not:",time,"; count or not:",count,"; normalise or not:",normalise)
                
            subdir=data_cat+'/'+name_reg[int(regression)]+'/'+str(minlen)+subdir_name
            data_subdir=DATA_processed+subdir
            X=load_pickle(data_subdir+"X_count_time_auged.pkl")
            name='_count_time_auged_'                        

            Sig_features=signature_featuring(X,order, time=time,start_replace=start_replace,leadlag=leadlag,rectilinear=rectilinear,log=log)
                        
            names=name+str(int(leadlag))+'_'+str(int(rectilinear))
            if log:
                 sig_name='logsig_'+str(order)
            else:
                sig_name='sig_'+str(order)
                                    
            np.save(data_subdir+sig_name+"_features"+names+".npy",Sig_features)
                                       
            print('Processing data-mean features without missing vaues')
            mean_features=mean_featuring(X)
            np.save(data_subdir+"mean_features"+name+".npy",mean_features)

                
            print('\n')
                
      
        
        
