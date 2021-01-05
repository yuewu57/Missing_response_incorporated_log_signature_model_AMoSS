import numpy as np


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score,roc_curve, f1_score
from sklearn.metrics import explained_variance_score,mean_squared_error, r2_score, median_absolute_error

from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.base import clone

from definitions import *
from src.models.plots import *

#####################For hyperparameter tuning: nested LOOCV ############################

lgbm_grid_parameters ={ # LightGBM
        'n_estimators': [40,70,100,200,400,500, 800],
        'learning_rate': [0.08,0.1,0.12,0.05],
        'colsample_bytree': [0.5,0.6,0.7, 0.8],
        'max_depth': [4,5,6,7,8],
        'num_leaves': [5,10,16, 20,25,36,49],
        'reg_alpha': [0.001,0.01,0.05,0.1,0.5,1,2,5,10,20,50,100],
        'reg_lambda': [0.001,0.01,0.05,0.1,0.5,1,2,5,10,20,50,100],
        'min_split_gain': [0.0,0.1,0.2,0.3, 0.4],
        'subsample': np.arange(10)[5:]/12,
        'subsample_freq': [10, 20],
        'max_bin': [100, 250,500,1000],
        'min_child_samples': [49,99,159,199,259,299],
        'min_child_weight': np.arange(30)+20}


logistic_grid_parameters ={ # Logistic regression
#     'C': [0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10,20,50]
    'C':np.logspace(-2, 1.5, 30)}


 

def nestedLOOCV_one_per_par(model, X, y,param_grid, reg=False,grid=False,n_iter=100, n_jobs=-1, scoring='roc_auc',verbose=2,one_rest=False):
    
    """
        
        For chosen base model, we conduct hyperparameter-tuning on given cv splitting.
        
        Input:
            model
            dataset: numpy version
            labels: numpy array
            tra_full_indices/val_full_indices: from cv splitting
            param_grid:for hyperparameter tuning
        
        Output:
            
        
        
    """    

    
    # evaluate model
    
    loocv=LeaveOneOut()
    if reg:
        test_true=np.empty((0,1),float)
        test_preds=np.empty((0,1),float)
    else:
        test_true=np.empty((0,1),int)
        test_preds=np.empty((0,1),int)
        prob_preds=np.empty((0,len(np.unique(y))),float)
    
    if one_rest:
        model=OneVsRestClassifier(model)
        
    for train_ix, test_ix in loocv.split(X):
        
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]

        
            cv_now=loocv.split(X_train)
        
            if grid:
                gs = GridSearchCV(estimator=model, \
                                  param_grid=param_grid,\
                                  n_jobs=n_jobs,\
                                  cv=cv_now,\
                                  scoring=scoring,\
                                  verbose=verbose)
            else:
        
                gs = RandomizedSearchCV(model, \
                                        param_grid,\
                                        n_jobs=n_jobs,\
                                        n_iter=n_iter,\
                                        cv=cv_now,\
                                        scoring=scoring,\
                                        verbose=verbose)  
            
            fitted_model=gs.fit(X=X_train,y=y_train)
            
            best_params_=fitted_model.best_params_
            print(best_params_)
            clf=model.set_params(**best_params_)

            # evaluate model
        
        
            test_true=np.append(test_true,y_test)
            clf.fit(X_train,y_train)
            if reg:
                test_preds=np.append(test_preds,clf.predict(X_test))
            
            else:
                
                test_probs = clf.predict_proba(X_test)
               
                prob_preds=np.concatenate((prob_preds,test_probs))
               
                test_preds=np.append(test_preds,np.where(test_probs[0]==np.max(test_probs[0]))[0][0])
                
    
    if reg:
        return test_true,test_preds
    else:
        return test_true,test_preds,prob_preds

    
def full_idx_creator(lens):
    
    cum_lens=np.insert(np.cumsum(lens),0,0)
    idxs=[np.arange(lens[i])+cum_lens[i] for i in range(len(lens))]
    return idxs

def full_idx_splitting_creator(idxs,train_idx,test_idx):
    
    test_output=[idxs[test_idx[i]] for i in range(len(test_idx))]
    train_output=[idxs[train_idx[i]] for i in range(len(train_idx))]
    
    return np.concatenate(train_output), np.concatenate(test_output)
    
def nestedLOOCV_many_per_par(model, X, y, lens,param_grid, reg=False,grid=False,n_iter=100, n_jobs=-1,\
                       scoring='roc_auc',verbose=2,one_rest=False):
    
    """
        
        For chosen base model, we conduct hyperparameter-tuning on given cv splitting.
        
        Input:
            model
            dataset: numpy version
            labels: numpy array
            tra_full_indices/val_full_indices: from cv splitting
            param_grid:for hyperparameter tuning
        
        Output:
            
            set of best parameters for base model
        
        
    """    

    
    # evaluate model
    
    loocv=LeaveOneOut()
    if reg:
        test_true=np.empty((0,1),float)
        test_preds=np.empty((0,1),float)
    else:
        test_true=[]
        prob_preds=np.empty((0,len(np.unique(y))),float)

    if one_rest:
        model=OneVsRestClassifier(model)    
    
    test_lens=[]
    for train_idx, test_idx in loocv.split(lens):
        
            train_ix, test_ix=full_idx_splitting_creator(full_idx_creator(lens),train_idx,test_idx)
            
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            
            test_lens.append(lens[test_idx[0]])
            
            lens_now=np.delete(lens,test_idx)
            cv_now=loocv.split(lens_now)
            
            cv=[list(full_idx_splitting_creator(full_idx_creator(lens_now),train_idx_now,test_idx_now))\
                                                                        for train_idx_now,test_idx_now in cv_now]
            
            if grid:
                gs = GridSearchCV(estimator=model, \
                                  param_grid=param_grid,\
                                  n_jobs=n_jobs,\
                                  cv=cv,\
                                  scoring=scoring,\
                                  verbose=verbose)
            else:
        
                gs = RandomizedSearchCV(model, \
                                        param_grid,\
                                        n_jobs=n_jobs,\
                                        n_iter=n_iter,\
                                        cv=cv,\
                                        scoring=scoring,\
                                        verbose=verbose)  
            
            fitted_model=gs.fit(X=X_train,y=y_train)
            
            best_params_=fitted_model.best_params_
            print(best_params_)
            clf=model.set_params(**best_params_)

            # evaluate model
        
        
            test_true.append(y_test)
            clf.fit(X_train,y_train)
            if reg:
                test_preds=np.append(test_preds,clf.predict(X_test))
            
            else:
                
                test_probs = clf.predict_proba(X_test)
               
                prob_preds=np.concatenate((prob_preds,test_probs))
               
                
                
    
    if reg:
        return test_true,test_preds
    else:
        test_preds=np.argmax(prob_preds,axis=1)
        return np.concatenate(test_true),test_preds,prob_preds,test_lens

############################# CV: LOOCV or k-fold #################################

def cv_scores(model,X, y, k=None, lens=None, reg=False, one_rest=False,scoring='accuracy'):
    
    if one_rest:
        model=OneVsRestClassifier(model)    
    
    if lens is not None:
        if k is not None:
            cvs=KFold(n_splits=k).split(lens)
        else:
            cvs=LeaveOneOut().split(lens)
    else:
        if k is not None:
            cvs==KFold(n_splits=k).split(X)
        else:
            cvs=LeaveOneOut().split(X)  
        
    if lens is not None:
            cvs=[list(full_idx_splitting_creator(full_idx_creator(lens),train_idx,test_idx))\
                      for train_idx, test_idx in cvs]
            
    return cross_val_score(model, X, y, cv=cvs,  scoring=scoring)
            
def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
    
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    print(benchmark_score)
    # list for storing feature importances
    importances = []
    
    col_num=X_train.shape[-1]
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for i in range(col_num):
        
        indices=np.delete(np.arange(col_num),i)
        X_train_drop=X_train[:,indices]
        print(indices)
        
#     for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train_drop, y_train)
        drop_col_score = model_clone.score(X_train_drop, y_train)
        print(drop_col_score)
        importances.append(benchmark_score - drop_col_score)
    
#     importances_df = imp_df(np.arange(col_num), importances)
#     return importances_df
    return np.array(importances)

from sklearn.inspection import permutation_importance


def cv_model(model, X, y, k=None, lens=None, id_list=None,reg=False,one_rest=False):
    
    """
        
        For chosen base model, we conduct hyperparameter-tuning on given cv splitting.
        
        Input:
            model
            dataset: numpy version
            labels: numpy array

        Output:
            
               
    """    
 
    # evaluate model
    
    if reg:
        test_true=np.empty((0,1),float)
        test_preds=np.empty((0,1),float)
    else:
        test_true=[]
        
        prob_preds=np.empty((0,len(np.unique(y))),float)

    if one_rest:
        model=OneVsRestClassifier(model)    
    
    
    if lens is not None:
        test_lens=[]
        if k is not None:
            cvs=KFold(n_splits=k).split(lens)
        else:
            cvs=LeaveOneOut().split(lens)
    else:
        if k is not None:
            cvs=KFold(n_splits=k).split(X)
        else:
            cvs=LeaveOneOut().split(X)    
    
    if id_list is not None:
        test_ids=[]
    
    feat_impt_list = []
    
    permuted_fi_list=[]
    permuted_fi_std_list=[]
    
    drop_fi_list=[]
    
    for train_ix, test_ix in cvs:
            
            if id_list is not None:
                test_ids+=[id_list[test_ix[i]] for i in range(len(test_ix))]
            
            if lens is not None:
                
                train_idx, test_idx=full_idx_splitting_creator(full_idx_creator(lens),train_ix,test_ix)
                
                test_lens.append(lens[test_ix])
                
                train_ix, test_ix=train_idx, test_idx
            
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
                      
            # evaluate model
        
            test_true.append(y_test)
            
            model.fit(X_train,y_train)

            
            feat_impts = [] 
            for clf in model.estimators_:
                feat_impts.append(clf.feature_importances_)

            feat_impts=np.array(feat_impts)
            feat_impts=np.mean(feat_impts, axis=0)
            feat_impt_list.append(feat_impts)
            
            print('Now permutation')
#             r=permutation_importance(model, X_train,y_train, n_repeats=10, random_state=0)
#             print('Done')
#             permuted_fi_list.append(r['importances_mean'])
#             permuted_fi_std_list.append(r['importances_std'])
            
#             drop_fi_list.append(drop_col_feat_imp(model, X_train, y_train))
            
            if reg:
                test_preds=np.append(test_preds,clf.predict(X_test))
            
            else:
                
                test_probs = model.predict_proba(X_test)
               
                prob_preds=np.concatenate((prob_preds,test_probs))
    
    feat_impt_list=np.array(feat_impt_list)
    feat_impt_list=np.mean(feat_impt_list, axis=0)
    indices = np.argsort(feat_impt_list)[::-1]
    print(indices)
    print('default fi',feat_impt_list[indices])
    
#     permuted_fi_list=np.array(permuted_fi_list)
#     permuted_fi_list=np.mean(permuted_fi_list, axis=0)
#     indices1 = np.argsort(permuted_fi_list)[::-1]
#     print(indices1)
#     print('permutation fi',permuted_fi_list[indices1])  
    
#     permuted_fi_std_list=np.array(permuted_fi_std_list)
#     permuted_fi_std_list=np.mean(permuted_fi_std_list, axis=0)
#     print(permuted_fi_std_list[indices1])  
    
#     drop_fi_list=np.array(drop_fi_list)
#     drop_fi_list=np.mean(drop_fi_list,axis=0)
#     indices2 = np.argsort(drop_fi_list)[::-1]
#     print('drop col fi',indices2)
#     print(drop_fi_list[indices2])    
    
    if reg:
        return test_true,test_preds
    else:
        test_preds=np.argmax(prob_preds,axis=1)
        
        if lens is not None:
            if id_list is not None:
                return np.concatenate(test_true),test_preds,prob_preds,np.concatenate(test_lens),np.array(test_ids,dtype='int')
            else: 
                return np.concatenate(test_true),test_preds,prob_preds,np.concatenate(test_lens)
        
        else:
            return np.concatenate(test_true),test_preds,prob_preds            

#################################### model performance ###############################################################

def model_performance_instance_level(test_true,test_preds,measurements,prob_preds=None,args=[None,2]):
    
    for measure in measurements:
        
        print(str(measure))
        print(measure(test_true,test_preds))
        
    if prob_preds is not None:
        
        if prob_preds.shape[1]>2:
            test_true=preprocessing.label_binarize(test_true, classes=np.arange(prob_preds.shape[1]))
        
            plot_roc(test_true, prob_preds, title=args[0],n_classes=prob_preds.shape[1],lw=args[1])
        else:
            
            binary_auc_plot(test_true, prob_preds[:,-1],save_name=args[0],lw=args[1])
            
def major_voting(a,probs=None):
    
    uniques,counts=np.unique(a,return_counts=True)
    
    if probs is not None:
        return uniques[np.argmax(counts)]
    else:
        mean_probs=np.mean(a,axis=0)
        max_count=counts[np.argmax(counts)]
        max_ids=np.where(counts==max_count)[0]
        
        if len(max_ids)>1:
            return uniques[max_ids[np.argmax(mean_probs[max_ids])]]
            
        else:
            return uniques[np.argmax(counts)]
        
def max_mean_probs(a):

    mean_probs=np.mean(a,axis=0)

    return np.argmax(mean_probs)




def model_performance_participant_level(test_true,test_preds,measurements,test_lens,test_ids=None,prob_preds=None,args=[None,2]):
    
    # for classification

    full_idxs=full_idx_creator(test_lens)
    test_true=np.array([test_true[full_idxs[i][0]] for i in range(len(test_lens))])
    test_preds1=np.array([major_voting(test_preds[full_idxs[i]],probs=prob_preds) for i in range(len(test_lens))])
        
    if prob_preds is not None: 
            test_preds2=np.array([max_mean_probs(prob_preds[full_idxs[i]]) for i in range(len(test_lens))])
    
    if test_ids is not None:
        
        correct_ids_measurements=[[] for i in range(len(measurements))]
        
    CMs=[[] for i in range(len(measurements))]
    
    measure_results=[[] for i in range(len(measurements))]
    for i in range(len(measurements)):
        
            measure=measurements[i]
            
            print(str(measure))
            measure_results[i].append(measure(test_true,test_preds1))
            print('Patient level (major voting):',measure_results[i][-1])
            
            CMs[i].append(confusion_matrix(test_true,test_preds1))
            
            if test_ids is not None:
                correct_ids=np.where((test_true-test_preds1)==0)[0]
                correct_ids_measurements[i].append(test_ids[correct_ids])
            
            if prob_preds is not None:
                measure_results[i].append(measure(test_true,test_preds2))
                print('Patient level (max mean_probs):',measure_results[i][-1])
                CMs[i].append(confusion_matrix(test_true,test_preds2))
                
                if test_ids is not None:
                    correct_ids=np.where((test_true-test_preds2)==0)[0]
                    correct_ids_measurements[i].append(test_ids[correct_ids])
                
            
        
    if prob_preds is not None:
        prob_preds=np.concatenate([np.array([np.mean(prob_preds[full_idxs[i]],axis=0)]) for i in range(len(test_lens))],axis=0)

        if prob_preds.shape[-1]>2:
            
            test_true=preprocessing.label_binarize(test_true, classes=np.arange(prob_preds.shape[1]))

            roc_aucs=plot_roc(test_true, prob_preds, title=args[0],n_classes=test_true.shape[1],lw=args[1])
        else:

            roc_aucs=binary_auc_plot(test_true, prob_preds[:,-1],save_name=args[0],lw=args[1])
            
    if test_ids is not None and prob_preds is not None:
        return measure_results, CMs,roc_aucs,correct_ids_measurements,test_preds1
    elif test_ids is not None and prob_preds is None:
        return measure_results, CMs,correct_ids_measurements,test_preds1
    elif test_ids is None and prob_preds is not None:
        return measure_results, CMs, roc_aucs,test_preds1       
    else:
        return measure_results, CMs,test_preds1
############################################
def delete_some_class(X,y,lens,ids,index=1,index_to_change=2):
    y=np.array(y)
    
    full_idxs=full_idxs=full_idx_creator(lens)
    y_par=np.array([y[full_idxs[i][0]] for i in range(len(lens))],dtype='int')
    
    idxs_to_keep=np.where(y_par!=index)[0]
    full_idxs_to_keep=np.concatenate([full_idxs[i] for i in range(len(lens)) if i in idxs_to_keep])

    lens_output=lens[idxs_to_keep]
    y_output=y[full_idxs_to_keep]
    X_output=X[full_idxs_to_keep,:]
    
    if index_to_change is not None:
        y_output[y_output==index_to_change]=index
        
    return X_output, y_output,lens_output,ids[idxs_to_keep]


def mean_feature(X,y,lens):
    
    y=np.array(y)
    
    full_idxs=full_idxs=full_idx_creator(lens)
    y_par=np.array([y[full_idxs[i][0]] for i in range(len(lens))],dtype='int')
    
    
    
    X_output=np.array([np.mean(X[full_idxs[i],:],axis=0) for i in range(len(lens))])
    

        
    return X_output, y_par

#####################################model for all features in one-go ##########################
def model_feature_many_per_par(model,names,data=None,k=3, minlen=10,\
                               subsubfolder='_general_group_md/',\
                               regression=False,one_rest=True,data_cat='weekly'):
    
        if regression:
            measurements=[explained_variance_score,  r2_score]
        else:
            measurements=[accuracy_score]
        

        
        print("Current data length:",minlen)
        
        
        data_subdir=DATA_processed+data_cat+'/'+name_reg[int(regression)]+'/'+str(minlen)+'/many_per_par'+subsubfolder
        print("Current data folder:",data_subdir)
        lens=np.load(data_subdir+'lens.npy')
        ids=load_pickle(data_subdir+'ids.pkl')
        y=load_pickle(data_subdir+'Y.pkl')
        
        if data is not None:
                               
               X=data                
        else:
            
                X=np.load(data_subdir+name)
        print(X.shape)



        test_true,test_preds,prob_preds,test_lens,test_ids=cv_model(model, X, np.array(y),\
                                                                        k=k,lens=lens,id_list=ids,\
                                                                        reg=regression,\
                                                                        one_rest=one_rest)

            
        model_performance_instance_level(test_true,test_preds,measurements,prob_preds=prob_preds,args=[None,2])
        accuracies,CMs,roc_aucs,correct_ids,test_par=model_performance_participant_level(test_true,test_preds,measurements,\
                                                     test_lens=test_lens,test_ids=test_ids,prob_preds=prob_preds,args=[None,2])

                
        print(CMs[0][0])
        print('\n')
                               
        return  prob_preds,test_lens,test_par,test_ids,correct_ids, accuracies,roc_aucs,X.shape[-1]
     

def model_features_many_per_par_in_one_go(model,names,k=3, minlen_list=[10,20],\
                                          subsubfolders=['_general_group_md/'],\
                                          regression=False,one_rest=True,data_cat='weekly'):
    
    if regression:
        measurements=[explained_variance_score,  r2_score]
    else:
        measurements=[accuracy_score]
    
    
    prob_list_list=[]
    test_len_list_list=[]
    test_preds_list_list=[]
    ids_list_list=[]
    correct_ids_list_list=[]
    results_list_list=[]
    roc_aucs_list_list=[]
    CMs_list_list=[]
    feature_len_list_list=[]
    test_true_list_list=[]
    
    for minlen in minlen_list: 
        
        prob_list=[]
        test_len_list=[]
        test_preds_list=[]
        test_true_list=[]
        ids_list=[]
        correct_ids_list=[]        
        results_list=[]
        roc_aucs_list=[]
        CMs_list=[]
        feature_len_list=[]
        
        print("Current data length:",minlen)
        
        for subsubfolder in subsubfolders:
            data_subdir=DATA_processed+data_cat+'/'+name_reg[int(regression)]+'/'+str(minlen)+'/many_per_par'+subsubfolder
            print("Current data folder:",data_subdir)
            lens=np.load(data_subdir+'lens.npy')
            ids=load_pickle(data_subdir+'ids.pkl')
            y=load_pickle(data_subdir+'Y.pkl')
        
            for name in names:
            
                X=np.load(data_subdir+name)
                print(X.shape)
                feature_len_list.append(X.shape[-1])
            
                print(name+':')
                X=np.load(data_subdir+name)

                test_true,test_preds,prob_preds,test_lens,test_ids=cv_model(model, X, np.array(y),\
                                                                        k=k,lens=lens,id_list=ids,\
                                                                        reg=regression,\
                                                                        one_rest=one_rest)

                prob_list.append(prob_preds)
                
                test_len_list.append(test_lens)
                ids_list.append(test_ids)
            
                model_performance_instance_level(test_true,test_preds,measurements,prob_preds=prob_preds,args=[None,2])
                accuracies,CMs,roc_aucs,correct_ids,test_par=model_performance_participant_level(test_true,test_preds,measurements,\
                                                     test_lens=test_lens,test_ids=test_ids,prob_preds=prob_preds,args=[None,2])
                correct_ids_list.append(correct_ids)
                results_list.append(accuracies)
                roc_aucs_list.append(roc_aucs)
                test_true_list.append(test_true)
                test_preds_list.append(test_par)
                CMs_list.append(CMs)
                
                print(CMs[0][0])
                print(CMs[0][-1])
                print('\n')
                
        prob_list_list.append(prob_list)
        test_len_list_list.append(test_len_list)
        test_preds_list_list.append(test_preds_list)
        ids_list_list.append(ids_list)
        correct_ids_list_list.append(correct_ids_list)
        results_list_list.append(results_list)
        roc_aucs_list_list.append(roc_aucs_list)
        CMs_list_list.append(CMs_list)
        feature_len_list_list.append(feature_len_list)
        test_true_list_list.append(test_true_list)
    
    return prob_list_list,test_len_list_list,test_preds_list_list,ids_list_list,\
           correct_ids_list_list, results_list_list,roc_aucs_list_list,feature_len_list_list,test_true_list_list