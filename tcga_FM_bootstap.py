import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from utils import featre_to_tick

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tqdm import tqdm
from  sklearn.preprocessing import StandardScaler


from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index as cindex
from  sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
USE_CUDA = torch.cuda.is_available() 
from torch.autograd import Variable

def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
def toTensor(v,dtype = torch.float,requires_grad = False):       
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))
def toNumpy(v):
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()
print('Using CUDA:',USE_CUDA)
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
class SurvRanker:
    def __init__(self,lambdaw=0.1,p=1,Tmax = 1000,lr=1e-2):#,lambdaw=0.010,p=1,Tmax = 100,lr=1e-1
        self.lambdaw = lambdaw
        self.p = p
        self.Tmax = Tmax
        self.lr = lr
        #return self
    def fit(self,X_train,T_train,E_train):        
        from sklearn.preprocessing import MinMaxScaler
        #self.MMS = MinMaxScaler().fit(T_train.reshape(-1, 1))#rescale y-values
        #T_train = 1e-3+self.MMS.transform(T_train.reshape(-1, 1)).flatten()        
        x = toTensor(X_train)
        y = toTensor(T_train)
        e = toTensor(E_train)
        N,D_in = x.shape
        H, D_out = D_in, 1
        model = torch.nn.Sequential(
            #torch.nn.Linear(D_in, H,bias=False),
            #torch.nn.Tanh(),   
            # torch.nn.Linear(H, H,bias=True),
            # torch.nn.Sigmoid(),  
            torch.nn.Linear(H, D_out,bias=False),
            #torch.nn.Tanh()
        )
        
        model=cuda(model)
        learning_rate = self.lr
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.0)
        TT = self.Tmax  
        lambdaw = self.lambdaw 
        p = self.p
        L = []
        dT = T_train[:, None] - T_train[None, :] #dT_ij = T_i-T_j
        dP = (dT>0)*E_train
        dP = toTensor(dP,requires_grad=False)>0 # P ={(i,j)|T_i>T_j ^ E_j=1}
        dY = (y.unsqueeze(1) - y)[dP]
        for t in (range(TT)):
            y_pred = model(x).flatten()
            dZ = (y_pred.unsqueeze(1) - y_pred)[dP]  
            loss = torch.mean(torch.max(toTensor([0],requires_grad=False),1.0-dZ)**p) #hinge loss
            
       
            
            ww = torch.cat([w_.view(-1) for w_ in model[0].parameters()]) #weights of input
            loss+=lambdaw*torch.norm(ww, p)**p #regularize
            L.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
        ww = torch.cat([w_.view(-1) for w_ in model[0].parameters()])
        self.ww = ww
        self.L = L
        self.model = model
        return self
    def decision_function(self,x):
        x = toTensor(x)
        return toNumpy(self.model(x))
    

BOOTSTRAP_RUNS = 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cancer_types', nargs='+', required=True)
    parser.add_argument('--cancer_subset', default=None)
    parser.add_argument('--event_type', required=True)
    parser.add_argument('--censor_at', type=int, default=-1)
    parser.add_argument('--results_root', default='./results_final/survival/FS_results/')

    args = parser.parse_args()

    cancer_types = args.cancer_types
    cancer_subset = args.cancer_subset
    event_type = args.event_type
    censor_at = args.censor_at
    results_root = args.results_root


    discov_val_feats_path = '/home/u2070124/lsf_workspace/Data/Data/pancancer/tcga_features_final.csv'
    # discov_val_feats_path = '/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_clinical_merged.csv'
    discov_df = pd.read_csv(discov_val_feats_path)
    discov_df = discov_df.loc[discov_df['type'].isin(cancer_types)]
    ff = pd.read_csv('noncorrolated_features_list_final.csv', header=None)[0].to_list()
    print("Using features: ", ff)
    print(f"Initial number of cases in this cancer type: {len(discov_df)}")
    
    nfolds = [1,2,3] # ALAKI
    model_type = 'cox' ## 'rsf': for Random Survival Forest, 'cox': for Cox PH regression model
    save_plot = False # ALAKI
    rsf_rseed = 100 ## random seed for Random Survival Forest
    cutoff_mode = 'median' ## 'median' | 'mean' ## the cut off point calculation for stratification of high vs low risk cases
    cutoff_point =  -1 ## 0.92 ##if set to -1 then median will be used as cut off. If set to any other positive value then cut_mode option will be ignores and the fixed cut off provided will be used for stratification of high vs low risk cases
    time_col = f'{event_type}.time'
    event_col = event_type
    subset = cancer_subset ##'Endocrine'|'Endocrine_LN0'; 'Endocrine': Endocrine treated only with lymph node 0-3; 'Endocrine_LN0': Endocrine treated lymph node negative
    
    discov_df = discov_df.dropna(subset=[event_col, time_col])
    discov_df[event_col] = discov_df[event_col].astype(int)
    discov_df[time_col] = (discov_df[time_col]/30.4).astype(int)

    print(f"Number of cases after dropping NA: {len(discov_df)}")
    
    save_dir = results_root + f"FM_{cancer_types}/"
    os.makedirs(save_dir, exist_ok=True)

    XX = np.array(discov_df[ff])
    TT = np.array(discov_df[time_col])
    EE = np.array(discov_df[event_col])

    if censor_at != -1:
        EE[TT>censor_at] = 0
        TT[TT>censor_at] = censor_at
    
    ######
    N = XX.shape[0]
    #print(dataset.shape)
    CC = []
    AA_ll= [];PP = []
    WW = []
    cphcindex = []
    cphHR = []; cph_PP = []
    use_cph = 1 #enable to use CPH for comparison
    permute_times = False 
    permute_events = False
    rng = np.random.RandomState()

    for _ in tqdm(range(BOOTSTRAP_RUNS)):
        #index_train, index_test = train_test_split( range(N), test_size = 0.4, shuffle=True,  stratify = EE)
        index_train = list(rng.choice(np.nonzero(EE==0)[0],size = len(EE)-np.sum(EE),replace=True))+list(rng.choice(np.nonzero(EE==1)[0],size = np.sum(EE),replace=True))
        index_test = list(set(range(len(EE))).difference(index_train))
        # Creating the X, T and E input
        X_train, X_test = XX[index_train], XX[index_test]
        T_train, T_test = TT[index_train], TT[index_test]
        E_train, E_test = EE[index_train], EE[index_test]
        
        MMS = StandardScaler().fit(X_train)
        X_train = MMS.transform(X_train)
        X_test = MMS.transform(X_test) 
        if use_cph:
            
            dftr = pd.DataFrame(dict(zip(ff+[time_col, event_col],np.hstack((X_train,T_train[...,np.newaxis],E_train[...,np.newaxis])).T)))
            dftt = pd.DataFrame(dict(zip(ff+[time_col, event_col],np.hstack((X_test,T_test[...,np.newaxis],E_test[...,np.newaxis])).T)))
            # dfall = dftr.append(dftt)
            dfall = pd.concat([dftr, dftt])
            try:
                cph = CoxPHFitter(penalizer = 0.001, l1_ratio = 0.5, baseline_estimation_method="breslow").fit(dftr, time_col, event_col)#,robust = True        
            except:
                print('Failed to converge')
                continue

            cphcindex.append(cindex(dftt[time_col], -cph.predict_partial_hazard(dftt), dftt[event_col]))
            cphHR.append(np.array(cph.hazard_ratios_))
            
            thr = np.mean(cph.predict_partial_hazard(dftr))
            Z_test = cph.predict_partial_hazard(dftt)
            results = logrank_test(T_test[Z_test>thr], T_test[Z_test<=thr], E_test[Z_test>thr],E_test[Z_test<=thr])
            cph_PP.append( results.p_value)

        
        if permute_events: E_test = np.random.permutation(E_test)
        if permute_times: T_test = np.random.permutation(T_test)

        sr = SurvRanker().fit(X_train,T_train,E_train)
        W=toNumpy(sr.ww);W = W/np.linalg.norm(W,ord=1)
        
        WW.append(W)

        Z_train = np.dot(X_train,W)
        Z_test = np.dot(X_test,W)
        c_index_ll = cindex(T_test, Z_test, E_test) #lifelines
        
        thr = np.median(Z_train)
        results = logrank_test(T_test[Z_test>thr], T_test[Z_test<=thr], E_test[Z_test>thr],E_test[Z_test<=thr])
        #results.print_summary()
        PP.append(results.p_value)
        AA_ll.append(c_index_ll)

    results_dcit = {'ml_cindex': AA_ll,
                    'ml_pvalue': PP,
                    'cph_cindex': cphcindex,
                    'cph_pvalue': cph_PP}
    results_df = pd.DataFrame(results_dcit)
    print('ML C-value mean/std',np.mean(AA_ll),np.std(AA_ll))
    print('ML 2p50',2*np.median(PP))
    print('CPH c-index',np.mean(cphcindex),np.std(cphcindex))
    print('CPH 2p50',2*np.median(cph_PP))
    WW = np.array(WW)

    ww = np.median(WW,axis=0)
    idx = np.argsort(-np.abs(ww));
    ffx =  np.array([' '.join(f.split('_')) for f in ff])
    ffx = np.array([featre_to_tick(f) for f in ff])

    save_dir = f'./{results_root}/FM_{cancer_types}/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir + f"FM_bootstrap_results_{cancer_types}_{event_type}_censor{censor_at}"
    results_df.to_csv(save_path+'.csv', index=None)

    plt.figure(); plt.boxplot(WW[:,idx]);plt.xticks(list(np.arange(1,len(ffx)+1)), np.array(ffx)[idx]);plt.xticks(rotation=90);plt.title(''.join(cancer_types).upper())
    plt.savefig(save_path+'_L1RM.png', dpi=600, bbox_inches = 'tight', pad_inches = 0)
    plt.savefig(save_path+'_L1RM.pdf', dpi=600, bbox_inches = 'tight', pad_inches = 0)
    plt.figure(); plt.violinplot(WW[:,idx[::-1]],showmedians=True,showextrema=True,vert=False,widths=0.9);plt.yticks(list(np.arange(1,len(ff)+1)), ffx[idx[::-1]]);plt.yticks(rotation=0);plt.title(''.join(cancer_types).upper());plt.xticks(fontsize=14);plt.yticks(fontsize=14);plt.xlabel('Weight',fontsize=16)
    plt.savefig(save_path+'_ML.png', dpi=600, bbox_inches = 'tight', pad_inches = 0)
    plt.savefig(save_path+'_ML.pdf', dpi=600, bbox_inches = 'tight', pad_inches = 0)
    if use_cph: plt.figure(); plt.violinplot(np.log(np.array(cphHR))[:,idx[::-1]],showmedians=True,showextrema=True,vert=False,widths=0.9);plt.yticks(list(np.arange(1,len(ff)+1)), ffx[idx[::-1]]);plt.yticks(rotation=0);plt.title(''.join(cancer_types).upper());plt.xticks(fontsize=14);plt.yticks(fontsize=14);plt.xlabel('log(HR)',fontsize=16)
    plt.savefig(save_path+'_COX.png', dpi=600, bbox_inches = 'tight', pad_inches = 0)
    plt.savefig(save_path+'_COX.pdf', dpi=600, bbox_inches = 'tight', pad_inches = 0)
