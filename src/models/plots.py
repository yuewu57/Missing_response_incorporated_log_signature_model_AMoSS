from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

colors_auc=sns.color_palette("Dark2")
linestyles=[':','-.','-','--']        
colors_shade=sns.color_palette("Pastel2")   


def plot_roc(y_test, y_score, title=None,n_classes=3,lw=2):
        
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

   # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(4,3))
    
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC ({0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC ({0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    
    
    for i, color in zip(range(n_classes), colors):
        classes = {0: "BPD", 1: "HC", 2: "BD"}[i]
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC of ' +classes+' ({1:0.2f})'
                 ''.format(i,roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
   # plt.title('Receiver operating characteristic for '+title)
 #   plt.legend(loc="lower right", bbox_to_anchor=(1.8, 0.5))
    plt.legend(loc="lower right")
    if title==None:
        plt.show()  

    else:
        plt.savefig('ROC_for_'+title+'.tiff',dpi=300)
    
    return roc_auc

def binary_auc_plot(trues,probs,fontsize=14,\
                    colors=colors_auc,linestyles=linestyles,\
                    lw = 2,loc="lower right", save_name=None):
    
    """
        AUC plots in one figure via ground truth and predicted probabilities
        
    Input:
    
        trues_list: ground-truth-seq list 
        
                eg, for 2 set of data, [[ground truth for set1],[ground truth for set2]]
                
        probs_list: probability-seq list
        
            eg, for 2 set of data, [[probabilities for set1],[probabilities for set2]]
            
        names: curve labels
        
        save_name: if None: print figure; else: save to save_name.png
        
    """
    

    
    plt.figure()

        
    fpr, tpr, _ = roc_curve(trues, probs)
    roc_auc = auc(fpr, tpr)
        
    plt.plot(fpr, tpr, color=colors[0],linestyle=linestyles[0],\
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.ylabel('True Positive Rate',fontsize=fontsize)
    plt.legend(loc=loc,fontsize=fontsize-3)
    
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    if save_name is not None:
        plt.savefig(save_name+'.png',dpi=300)
    else:        
        plt.show()
    
    return roc_auc
    
def binary_aucs_plot(trues_list,probs_list,names,fontsize=14,\
                    colors=colors_auc,linestyles=linestyles,\
                    lw = 2,loc="lower right", save_name=None):
    
    """
        AUC plots in one figure via ground truth and predicted probabilities
        
    Input:
    
        trues_list: ground-truth-seq list 
        
                eg, for 2 set of data, [[ground truth for set1],[ground truth for set2]]
                
        probs_list: probability-seq list
        
            eg, for 2 set of data, [[probabilities for set1],[probabilities for set2]]
            
        names: curve labels
        
        save_name: if None: print figure; else: save to save_name.png
        
    """
    
    num=len(trues_list)
    
    plt.figure()
    roc_aucs=[]
    for i in range(num):
        
        fpr, tpr, _ = roc_curve(trues_list[i], probs_list[i])
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        plt.plot(fpr, tpr, color=colors[i],linestyle=linestyles[i],\
                 lw=lw, label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
        
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.ylabel('True Positive Rate',fontsize=fontsize)
    plt.legend(loc=loc,fontsize=fontsize-3)
    
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    if save_name is not None:
        plt.savefig(save_name+'.png',dpi=300)

    else:        
        plt.show()
    
    return roc_aucs




def CI_AUC_bootstrapping(n_bootstraps, alpha, y_true, y_pred, rng_seed = 1):
    # to compute alpha % confidence interval using boostraps for n_boostraps times 
    bootstrapped_scores = []
    fprs,tprs=[],[]
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        #sample_index = np.random.choice(range(0, len(y_pred)), len(y_pred))
       # print(indices)

        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
                continue
            

        score = roc_auc_score(y_true[indices], y_pred[indices])
        fpr, tpr, _= roc_curve(y_true[indices],y_pred[indices])
        fprs.append(fpr)
        tprs.append(tpr)
        bootstrapped_scores.append(score)
#         if i%20 ==0:
#             print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
            
    factor =  norm.ppf(alpha)
    std1 = np.std(bootstrapped_scores)
    mean1 = np.mean(bootstrapped_scores)
    up1 = mean1+factor*std1
    lower1 =  mean1-factor*std1
#     print( '{}% confidence interval is [{},{}]'.format(alpha, up1, lower1))
    return [lower1,up1],fprs,tprs

def fprs_tprs_output(labels,probs,n_bootstraps=100,alpha=0.95):


    
        fprs_list=[]
        tprs_list=[]   
    
        for j in range(3):
        
            CI_results,fprs,tprs=CI_AUC_bootstrapping(n_bootstraps, alpha,labels[:, j], probs[:, j],  rng_seed = 1)
        
            print(j,"{:.3f}".format(roc_auc_score(labels[:, j], probs[:, j])),\
              "["+"{:.3f}".format(CI_results[0]) +","+"{:.3f}".format(CI_results[1])+"]")

        
            fprs_list.append(fprs)
            tprs_list.append(tprs)

        print('\n')
        
        return fprs_list, tprs_list

def CI_std_output(fprs_list,tprs_list, mean_fpr_list=[np.linspace(0, 1, 100+0*i) for i in range(3)]):

        error_list=[]
        print('3:',len(mean_fpr_list[0]))
        for j in range(len(mean_fpr_list)):
            tprs_=[]
        
            for k in range(len(tprs_list[j])):
            
                fpr_now=fprs_list[j][k]
                tpr_now=tprs_list[j][k]
                interp_tpr = np.interp(mean_fpr_list[j], fpr_now, tpr_now)

                interp_tpr[0] = 0.0
                tprs_.append(interp_tpr)
        
            mean_tpr = np.mean(tprs_, axis=0)
            
            mean_tpr[-1] = 1.0
            
            std_tpr = np.std(tprs_, axis=0)
            print('last',std_tpr.shape)
            error_list.append(std_tpr)
            
        return error_list




def aucplot_errorbars(trues_list, probs_list,error_list,names,\
                      mean_fpr_list=[np.linspace(0, 1, 100+0*i) for i in range(3)],\
                      fontsize=14, figsize=(5,5),colors=colors_shade,colors_line=colors_auc,\
                      linestyles=linestyles,lw=2, loc="lower right",save_name=None):
    
    """
        
        AUC plots for different models via ground truth and predicted probabilities
        
    Input:
    
        trues_list: list of ground-truth-seq lists 
        
                eg, for three models for 2 set of data,[[model1 truth-list], [model2 truth-list], [model3 truth-list]]
                    [model1 truth-list]=[[ground truth for set1],[ground truth for set2]]
                
        probs_list: probability-seq list
        
            eg, for three models for 2 set of data,  [[model1 probs-list], [model2 probs-list], [model3 probs-list]]
            
                [model1 probs-list]=[[probabilities for set1],[probabilities for set2]]
            
        names: curve labels for sets of data
        
        save_name: if None: print figure; else: save to save_name.png

        
    
    """

    plt.figure(figsize=figsize)
    print('2:',len(mean_fpr_list[0]))
    num=trues_list.shape[-1]

    for i in range(num)[::-1]:
        
        fpr, tpr, _ = roc_curve(trues_list[:,i], probs_list[:,i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors_line[i],linestyle=linestyles[i],\
                 lw=lw, label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
        
        mean_tpr=np.interp(mean_fpr_list[i],fpr,tpr)
#         plt.errorbar(mean_fpr_list[i], mean_tpr, error_lists[i], color=colors[i],\
#                      linestyle=linestyles[i],label='ROC curve for '+names[i] +' (area = %0.2f)' % roc_auc)
        
        plt.fill_between(mean_fpr_list[i], mean_tpr-error_list[i], mean_tpr+error_list[i], color=colors[i])


    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.ylabel('True Positive Rate',fontsize=fontsize)
    plt.legend(loc=loc,fontsize=fontsize-3)
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-3)
    
    
    if save_name is not None:
        plt.savefig(save_name+'.jpeg',dpi=350)
    else:
        
        plt.show()

def prob_individual(probs):
    score=np.zeros(probs.shape[-1])
    preds=np.argmax(probs,axis=1)
    for i in range(len(score)):
        score[i]+=len(np.where(preds==i)[0])
    return score/len(preds)

def rocci_plot(test_true,test_lens,prob_preds,names=['BPD','HC','BD'],\
               n_bootstraps=100,alpha=0.95,args=[None,2],vote='hard',\
               mean_fpr_list=[np.linspace(0, 1, 100+0*i) for i in range(3)]):
    
    full_idxs=full_idx_creator(test_lens)
    if vote=='hard':
        prob_preds=np.concatenate([np.array([prob_individual(prob_preds[full_idxs[i]])]) for i in range(len(test_lens))],axis=0)
                                    
    else:    
        prob_preds=np.concatenate([np.array([np.mean(prob_preds[full_idxs[i]],axis=0)]) for i in range(len(test_lens))],axis=0)
    test_true=np.array([test_true[full_idxs[i]][0] for i in range(len(test_lens))])
    test_true=preprocessing.label_binarize(test_true, classes=np.arange(prob_preds.shape[1]))
    
    fprs_list, tprs_list=fprs_tprs_output(test_true,prob_preds,n_bootstraps=n_bootstraps,alpha=alpha)
    error_list=CI_std_output(fprs_list,tprs_list,mean_fpr_list=mean_fpr_list)

    aucplot_errorbars(test_true, prob_preds,error_list,names,save_name=args[0],mean_fpr_list=mean_fpr_list,lw=args[-1])
        
def plot_traj(idNumber,trun=None,fontsize=14,figsize=(10,3),\
              Qs=['ASRM','QIDS','EQ-5D','GAD-7'],\
              ylims=None,linestyle="steps-pre",lw=2,save=True):

    for par in weekly_data:
        if par.idNumber==idNumber:
            for j in range(len(par.data)):
                
                
                ##mask -1
                ym=np.ma.masked_where(par.data[j] < 0, par.data[j])
                y = par.data[j].copy()
                y[ym <0] = np.nan
                ####
                
                if trun is not None:
                    y=y[:trun]
                    
                plt.figure(figsize=figsize)
                
                if linestyle is not None:
                    plt.plot(np.arange(len(y)+1)[1:],y,label=Qs[j],linestyle=linestyle,lw=lw)
                else:
                    plt.plot(np.arange(len(y)+1)[1:],y,label=Qs[j],lw=lw)
                    
                plt.scatter(np.arange(len(y)+1)[1:],y)
                plt.legend(fontsize=fontsize)
                plt.xlabel('Week',fontsize=fontsize)
                plt.ylabel('Score',fontsize=fontsize)
                plt.xticks(fontsize=fontsize-2)
                plt.yticks(fontsize=fontsize-2)
                
                if ylims is not None:                    
                    plt.ylim(ylims[j])
                    
#                 plt.grid()
                if save:
                    plt.savefig(DATA_DIR+'plots/'+str(idNumber)+'_'+Qs[j]+'.jpeg',dpi=300,bbox_inches='tight')                    
                else:

                    plt.show()
                    
def plot_confusions(cm, target_names,figsize=(4,3), fontsize=14, savetitle=None):


    fig, ax = plt.subplots(figsize=figsize) 
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
    df_cm1 = pd.DataFrame(cm, target_names,
                  target_names)
    sns.set(font_scale=1.0)#for label size
    sns.heatmap(df_cm1, cmap="Blues", cbar=False, annot=True,annot_kws={"size": fontsize},fmt='g',ax=ax)# font size

    
    
    if savetitle==None:
        plt.show()        
    else:    
        plt.savefig(savetitle+'.jpeg',dpi=300)   
    
 
    
 
