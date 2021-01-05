"""
This file contains basic variables and definitions that we wish to make easily accessible for any script that requires
it.

from definitions import *
"""
from pathlib import Path

# Packages/functions used everywhere
from src.omni.functions import *


ROOT_DIR = str(Path(__file__).resolve().parents[0])

DATA_DIR = ROOT_DIR + '/data/'
MODELS_DIR = ROOT_DIR + '/models/'
OUTPUT_DIR=ROOT_DIR + '/outputs/'


DATA_processed=DATA_DIR + 'processed/'
DATA_raw=DATA_DIR + 'raw/'
DATA_interim=DATA_DIR + 'interim/'

exclude_ids=[14097,14037,14039]
may_exclude_ids=[14031,14117,14130]

data_cats=['aligned','weekly']
name_reg=['cla','reg']

def sig_data_dir(minlen,data_cat='aligned',regression=False,time=True,count=True,time_=True,leadlag=True):
    
    
    subdir=data_cat+'/'+name_reg[int(regression)]+'/'+str(minlen)+'/'
    data_subdir=DATA_processed+subdir
    
    if time:
                    name='_time_auged_'
    else:    
                    name='_'
    
    names=name+str(int(count))+'_'+str(int(time_))+'_'+str(int(leadlag))
        
    return data_subdir+"sig_features"+names+".pkl"
    
    
class Participant:
    
        def __init__(self, data, time, id_n,diagnosis,nextdata):

            self.idNumber = id_n
            self.data = data
            self.time=time
            self.diagnosis=diagnosis
            self.nextdata=nextdata

            
