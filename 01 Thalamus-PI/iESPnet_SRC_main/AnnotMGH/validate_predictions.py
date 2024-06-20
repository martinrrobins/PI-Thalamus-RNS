# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 08:31:12 2021

@author: vp820
"""

# check predictions
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np 
font = {'size'   : 14}
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, mean_absolute_error, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'utilities')))
import IO
import Epochs
import pandas as pd
matplotlib.rc('font', **font)
import seaborn as sb
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

#%%
prediction_path = 'X:/RNS_DataBank/MGH/MGH'
DATA_DIR = "X:/RNS_DataBank/MGH/MGH/"
mgh_subject_info = 'X:/iESPnet/Data/Metadatafiles/subjects_info_zeropadall_mgh.csv'
#%%
RNSIDS_MGH = IO.get_subfolders(DATA_DIR)

#get patient group to avoid validation in thalamic patients
mgh_df_subjects = pd.read_csv(mgh_subject_info)  

# keep annoted patients
RNSids_all_MGH = 'MGH-' + mgh_df_subjects.rns_id_np
idx_match = [idx for idx, s in enumerate(RNSids_all_MGH) if s in RNSIDS_MGH ] # ask for a and b in name
RNSIDS_MGH_b = [idd + 'b' for idd in RNSIDS_MGH]
idx_match_b = [idx for idx, s in enumerate(RNSids_all_MGH) if s in RNSIDS_MGH_b] # ask for a and b in name
idx_ = list(set(idx_match).union(set(idx_match_b)))

mgh_df_subjects_match = mgh_df_subjects.iloc[idx_,:]
# clean thalamus group
idx_nonthalamus =  [idx for idx, s in enumerate(mgh_df_subjects_match.group) if "thalamus" not in s ]
mgh_df_subjects_match_nothalamus = mgh_df_subjects_match.iloc[idx_nonthalamus,:]
RNSIDS_MGH = ('MGH-' + mgh_df_subjects_match_nothalamus.rns_id_np).tolist()
# remove b
RNSIDS_MGH = [ss.strip('b') for ss in RNSIDS_MGH] 
#%%
def plot_confusion_matrix(cm,
                          target_names,
                          accuracy,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import numpy as np
    import itertools

    # accuracy = np.trace(cm) / float(np.sum(cm))
    # misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(7.7, 6.1))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title+ '\n Balanced accuracy={:0.2f}'.format(accuracy))
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#%%
pp = PdfPages('AIannotation_validation_expert.pdf')

BAC = []
MAE = []
Label_expert_all = []
Label_ia_all = []
Time_ia_all = []
Time_expert_all = []
# now run per suject
for s in range(len(RNSIDS_MGH)):
    Label_expert = []
    Label_ia = []
    Time_ia = []
    Time_expert = []
    print('Running eval for subject ' + RNSIDS_MGH[s] + ' [s]: ' + str(s))

    annot_files = IO.get_annot_files(DATA_DIR, RNSIDS_MGH[s], 
                                 Verbose=False)

    IAannot_files = IO.get_annot_IA_files(prediction_path, RNSIDS_MGH[s])
    
    # where the annot files are willing to be saved
    save_predictions = prediction_path + RNSIDS_MGH[s] + '/' + 'iEEG'
    for nfile, annot_file in enumerate(annot_files):
        host, subj, PE = IO.get_patient_PE(annot_file, magicword = 'iEEG/')
        PE = PE[:10]
                    
        IAannot_file = [f for f in IAannot_files if PE in f][0]
        
        # get events accordingly to the net
        events_ia =  Epochs.get_events(IAannot_file, Verbose=False)
        
        # get events accordingly to expert
        events_expert =  Epochs.get_events(annot_file, Verbose=False)
        
        # read annot done by the net       
        annot_IA = np.loadtxt(IAannot_file, delimiter=',', skiprows=1,
                    dtype=str)
        # read annot done by the expert
        annot_VZ = np.loadtxt(annot_file, delimiter=',', skiprows=1,
                    dtype=str)
        
        # labels
        label_expert, time_expert = Epochs.get_segments_events(events_expert)
        label_ia, time_ia = Epochs.get_segments_events(events_ia)
        
        Label_expert.append(label_expert)
        Label_ia.append(label_ia)
        Time_expert.append(time_expert)
        Time_ia.append(time_ia)
        
    Label_expert =  np.concatenate(Label_expert)
    Label_ia = np.concatenate(Label_ia)
    Time_expert = np.concatenate(Time_expert)
    Time_ia = np.concatenate(Time_ia)
    
    Label_expert_all.append(label_expert)
    Label_ia_all.append(label_ia)
    Time_expert_all.append(time_expert)
    Time_ia_all.append(time_ia)
   

    del label_expert, time_expert, label_ia, time_ia   
    # BAC
    bac = balanced_accuracy_score(Label_expert, Label_ia)


    # MAE
    idx_tp = np.where((Label_expert == 1) & (Label_ia == 1))[0]
    if len(idx_tp) ==0:
        mae = None
    else:

        mae = mean_absolute_error(Time_expert[idx_tp], Time_ia[idx_tp])

    # to save on table
    BAC.append(bac)
    MAE.append(mae)
        
 
Label_expert_all =  np.concatenate(Label_expert_all)
Label_ia_all = np.concatenate(Label_ia_all)
Time_expert_all = np.concatenate(Time_expert_all)
Time_ia_all = np.concatenate(Time_ia_all)
#%%
cm = confusion_matrix(Label_expert_all, Label_ia_all)
# print(cm)
bac = balanced_accuracy_score(Label_expert_all, Label_ia_all)

plot_confusion_matrix(cm, normalize = False, 
                         target_names = ['no_iESP', 'iESP'],
                         accuracy= bac,
                         title        = ' ')
pp.savefig(axis='tight')

#$$
df_all = pd.DataFrame()
df_all['Patient'] = RNSIDS_MGH
df_all['Group'] = mgh_df_subjects_match_nothalamus.group.tolist()
df_all['balanced accuracy'] =  BAC
df_all['mean absolute error'] = MAE

plt.figure(figsize=(8,6))
plt.subplots_adjust(right=0.56)
g = sb.jointplot(data=df_all, x='mean absolute error', y='balanced accuracy', palette = 'Paired')
ax = sb.scatterplot(data=df_all, x='mean absolute error', y='balanced accuracy', hue='Group', palette = 'Paired', s=100, ax= g.ax_joint)
# ax = sb.stripplot(x="indices", y="F1",data=df_all,palette="rocket", dodge=False, s=5, alpha=0.9)

handles, labels = ax.get_legend_handles_labels()
l = plt.legend(handles, labels, loc='lower left', borderaxespad=0., ncol=1, bbox_to_anchor=(1.05, .77))    
ax.grid(color='gray', linestyle='--', linewidth=0.5, axis='y')
ax.grid(color='gray', linestyle='--', linewidth=0.5, axis='x')
# plt.axvline(x=10, color='k', linestyle='--')
# plt.ayhline(y=0.7, color='k', linestyle='--')
ax.set_ylim([0.5, 1.1])
ax.set_xlabel('Mean Absolute Error [s]');
pp.savefig(axis='tight')

   
plt.figure(figsize=(3,4))
ax =sb.boxplot(y="balanced accuracy",   
            data=df_all, palette="rocket", 
            showmeans=True, boxprops=dict(alpha=0.5), showcaps=True, showbox=True, 
            showfliers=False, notch=False,
            whiskerprops={'linewidth':2, "zorder":10, "alpha":0.5},
            capprops={"alpha":0.5},
            medianprops=dict(linestyle='-', linewidth=4, color="black", alpha=0.5))
ax = sb.stripplot(y="balanced accuracy",  data=df_all, palette="rocket", dodge=False, s=5, alpha=0.9)
ax.set_ylabel('balanced accuracy');
plt.subplots_adjust(left=0.28)
    
pp.savefig(axis='tight')


plt.figure(figsize=(3,4))
ax =sb.boxplot(y="mean absolute error",   
            data=df_all, palette="rocket", 
            showmeans=True, boxprops=dict(alpha=0.5), showcaps=True, showbox=True, 
            showfliers=False, notch=False,
            whiskerprops={'linewidth':2, "zorder":10, "alpha":0.5},
            capprops={"alpha":0.5},
            medianprops=dict(linestyle='-', linewidth=4, color="black", alpha=0.5))
ax = sb.stripplot(y="mean absolute error",  data=df_all, palette="rocket", dodge=False, s=5, alpha=0.9)
ax.set_ylabel('mean absolute error');
plt.subplots_adjust(left=0.28)
pp.savefig(axis='tight')


#%%
idx_tp = np.where((Label_expert_all == 1) & (Label_ia_all == 1))[0]   
df_time = pd.DataFrame()
df_time['True onset'] = Time_expert_all[idx_tp]
df_time['Predicted onset'] = Time_ia_all[idx_tp]
#%%rang

plt.figure(figsize=(5,4))
ax = sb.histplot(data=df_time, bins=np.arange(0,91,2), alpha=0.4) 
ax.set_xlabel('iESP onset time');
plt.subplots_adjust(left=0.2)
# plt.axvline(x=60, color='k', linestyle='--')
plt.subplots_adjust(bottom=0.2)
plt.xticks(np.arange(0, 91, step=10))
pp.savefig(axis='tight')

#%%
df_time['True'] = np.sort(Time_expert_all[idx_tp])
df_time['Predicted'] = np.sort(Time_ia_all[idx_tp])
df_time['Segment'] = range(len(idx_tp))

diff_time = Time_expert_all[idx_tp] - Time_ia_all[idx_tp]
plt.figure()
ax = sb.histplot(diff_time, alpha=0.5)
plt.figure()
ax =  sb.scatterplot(data=df_time, x= 'Segment', y='True' )
ax =  sb.scatterplot(data=df_time, x= 'Segment', y='Predicted' )
ax.set_ylabel('Onset time');
plt.legend(['True', 'Predicted'])
# plt.savefig(adress_tosave + 't.png')
# plt.savefig(adress_tosave + 'mae_all.svg')
#%%
x = Time_ia_all[idx_tp]
y = Time_expert_all[idx_tp]
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
plt.figure(figsize=(4.,3.))

g= sb.jointplot(data=df_time, y="True onset", x="Predicted onset", kind="reg",  height=4, ratio=2, color='royalblue')
g.ax_marg_x.set_axis_off()
g.ax_marg_y.set_axis_off()
if len(x) < 2:
    r, p = 0, 1
else:
    r, p = stats.pearsonr(x, y)
g.ax_joint.annotate(f'$\\rho = {r:.3f}, p < {0.0001}$',
                    xy=(0.1, 1), xycoords='axes fraction',
                    ha='left', va='center',
                    )
g.ax_joint.scatter(x, y)
g.set_axis_labels(xlabel='Predicted onset', ylabel='True onset', size=15)
plt.tight_layout()
plt.show()
pp.savefig(axis='tight')
