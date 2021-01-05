import sys
sys.path.insert(0, '../../')

from definitions import *
from src.features.preprocessing_functions.data_loading import *
from src.features.preprocessing_functions.class_making import *
from src.features.preprocessing_functions.data_cleaning import *


if __name__ == '__main__':

#     path='mat_files/'
#     print("Loading and saving data from path:",path)
#     loadParticipants_general(path)
#     print("Making classes for data and saving in",DATA_interim)
#     make_classes_general()
    
#     participants=load_pickle(DATA_interim+'participants_class_general.pkl')
    
#     rule1,rule2,rule3='average','union','first'
    
#     print("Cleaning same day data using rule ",rule1," and aligning paired data using rule ",rule2)
#     print("Saving first data for analysis after aligning data and excluding daily reported data, i.e., ids: ", exclude_ids)
#     participants_aligned=aligning_data_general(cleaning_same_day_data(participants,rule=rule1),rule=rule2)
    
#     print("Cleaning same calendar week data using rule ",rule3," and saving as the second data for analysis")

#     cleaning_sameweek_data_general(participants_aligned,rule=rule3,save=True)

    subdirs=["participants_class_aligned_general.pkl","participants_class_weekly_general.pkl"]
    for subdir in subdirs:
        clearning_redundant_missing_data(subdir,save=True)