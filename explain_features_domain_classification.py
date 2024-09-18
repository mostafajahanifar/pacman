import os
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import get_colors_dict, featre_to_tick
import pickle

save_root = 'results_final/landscape/domain_classification_one-vs-rest/'
os.makedirs(save_root, exist_ok=True)
# discov_val_feats_path = '/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_clinical_merged.csv'
discov_val_feats_path = '/home/u2070124/lsf_workspace/Data/Data/pancancer/tcga_features_final.csv'
discov_df = pd.read_csv(discov_val_feats_path)
feats_list = pd.read_csv('noncorrolated_features_list_final.csv', header=None)[0].to_list()
# feats_list = [feat for feat in feats_list if feat not in ["mit_clusterCoff_max", "mit_hotspot_score"]]

df = discov_df[['type']+feats_list]
invalid_cancers = ['UVM', 'LAML'] #['MESO', 'UVM', 'TGCT', 'THYM', 'THCA', 'LAML', 'DLBC', 'UCS', 'SARC', 'CHOL', 'PRAD', 'ACC'] # with kept PCPG
df = df[~df['type'].isin(invalid_cancers)]
df['type'] = df['type'].replace(['COAD', 'READ'], 'COADREAD')
df['type'] = df['type'].replace(['GBM', 'LGG'], 'GBMLGG')
domains = df['type'].unique()

# Separate features and target
X = df[feats_list]
y = df['type']


coefs_dict = {domain: {feat: [] for feat in feats_list} for domain in domains}
# features_dict = {feat: [] for feat in feats_list}


# Standardize the features
scaler = StandardScaler()

# Initialize the classifier
clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'), n_jobs=16)

# Number of bootstrap iterations
n_iterations = 500

# Store the AUROC for each bootstrap iteration
aucs_dict = {domain: [] for domain in domains}
f1_dict = {domain: [] for domain in domains}

print(feats_list)
print(len(feats_list))
for i in tqdm(range(n_iterations)):

    # Create train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Compute AUCs for each domain performance separately and save it into the aucs_dict
    y_score = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    for j, domain in enumerate(clf.classes_):
        y_true = (y_test == domain).astype(int)
        y_pred_domain = (y_pred == domain).astype(int)
        f1_dict[domain].append(f1_score(y_true, y_pred_domain))
        aucs_dict[domain].append(roc_auc_score(y_true, y_score[:, j]))

    # Collect the coefficient of estimators related to each domain
    for j, estimator in enumerate(clf.estimators_):
        for h, est_coef in enumerate(estimator.coef_.tolist()[0]):
            coefs_dict[clf.classes_[j]][feats_list[h]].append(est_coef)

# save_results to disk
with open(f'{save_root}/bootstrap_coefs_data.pkl', 'wb') as handle:
    pickle.dump(coefs_dict, handle)

# save_results to disk
with open(f'{save_root}/bootstrap_auc_data.pkl', 'wb') as handle:
    pickle.dump(aucs_dict, handle)

# save_results to disk
with open(f'{save_root}/bootstrap_f1_data.pkl', 'wb') as handle:
    pickle.dump(f1_dict, handle)

# plotting classification performance per domain
gap = 10
width = 8
color_dict = get_colors_dict()

fig = plt.figure(figsize=(15,3))  # Increase the figure size
ax = fig.add_subplot(111)
for i, domain in enumerate(domains):
    box1 = ax.boxplot(aucs_dict[domain], positions=[i*gap], widths=width, patch_artist=True, showfliers=False, boxprops=dict(facecolor=color_dict[domain]), medianprops=dict(color='black'))
ax.set_xticks(range(0, len(domains) * gap, gap))
ax.set_xticklabels(domains)
plt.xlim([-1*width, i*gap+1*width])
# Set the title and labels
plt.title(f'Domain Classification Performance')
plt.xlabel('Domains')
plt.ylabel('AUROC')
plt.savefig(f"{save_root}/Domain_Classification_Performance_AUC.png", bbox_inches='tight', dpi=600)

fig = plt.figure(figsize=(15,3))  # Increase the figure size
ax = fig.add_subplot(111)
for i, domain in enumerate(domains):
    box1 = ax.boxplot(f1_dict[domain], positions=[i*gap], widths=width, patch_artist=True, showfliers=False, boxprops=dict(facecolor=color_dict[domain]), medianprops=dict(color='black'))
ax.set_xticks(range(0, len(domains) * gap, gap))
ax.set_xticklabels(domains)
plt.xlim([-1*width, i*gap+1*width])
# Set the title and labels
plt.title(f'Domain Classification Performance')
plt.xlabel('Domains')
plt.ylabel('F1 Score')
plt.savefig(f"{save_root}/Domain_Classification_Performance_F1.png", bbox_inches='tight', dpi=600)

# For each feature
coefs_df = pd.DataFrame(coefs_dict).T
coefs_df.to_csv(f'{save_root}/bootstrap_coefs_data.csv')
