import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

save_root = 'explain_features/domain_classification_one-vs-one/'
os.makedirs(save_root, exist_ok=True)
discov_val_feats_path = '/home/u2070124/lsf_workspace/Data/Data/pancancer/tcga_features_clinical_merged.csv'
discov_df = pd.read_csv(discov_val_feats_path)
# discov_df = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_clinical_merged.csv')
feats_list = pd.read_csv('noncorrolated_feature_list_2.csv', header=None)[0].to_list()
feats_list = [feat for feat in feats_list if feat not in ["mit_clusterCoff_max", "mit_hotspot_score"]]

df = discov_df[['type'] + feats_list]
invalid_cancers = ['MESO', 'UVM', 'TGCT', 'THYM', 'THCA', 'LAML', 'DLBC', 'UCS', 'SARC', 'CHOL', 'PRAD', 'ACC']  # with kept PCPG
df = df[~df['type'].isin(invalid_cancers)]
df['type'] = df['type'].replace('COAD', 'COADREAD')
df['type'] = df['type'].replace('READ', 'COADREAD')
domains = df['type'].unique()

# Separate features and target
X = df[feats_list]
y = df['type']

# Standardize the features
scaler = StandardScaler()

# Initialize the classifier
clf = OneVsOneClassifier(svm.SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'), n_jobs=16)

# Number of bootstrap iterations
n_iterations = 500

# Store the AUROC for each bootstrap iteration
aucs_dict = {f'{domain1}_vs_{domain2}': [] for domain1 in domains for domain2 in domains if domain1 != domain2}

# Create a dictionary to map each pair to its index in the decision function output
pairwise_idx = {}
idx = 0
for i, domain1 in enumerate(domains):
    for j, domain2 in enumerate(domains):
        if i < j:  # Ensure each pair is only considered once
            pairwise_idx[(domain1, domain2)] = i # idx
            idx += 1

for i in tqdm(range(n_iterations)):

    # Create train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Compute AUCs for each pairwise domain performance and save it into the aucs_dict
    y_score = clf.decision_function(X_test)
    
    for (domain1, domain2), pair_idx in pairwise_idx.items():
        binary_idx = (y_test == domain1) | (y_test == domain2)
        y_test_binary = y_test[binary_idx]
        y_score_binary = y_score[binary_idx, pair_idx]
        
        y_true = (y_test_binary == domain1).astype(int)
        aucs_dict[f'{domain1}_vs_{domain2}'].append(roc_auc_score(y_true, y_score_binary))

# Save results to disk
with open(f'{save_root}/bootstrap_auc_data.pkl', 'wb') as handle:
    pickle.dump(aucs_dict, handle)

# Prepare data for heatmap
domain_pairs = list(aucs_dict.keys())
domain_aucs = {domain: [] for domain in domains}
for pair in domain_pairs:
    domain1, domain2 = pair.split('_vs_')
    if len(aucs_dict[pair])>0:
        mean_auc = np.mean(aucs_dict[pair])
        domain_aucs[domain1].append((domain2, mean_auc))
        domain_aucs[domain2].append((domain1, mean_auc))

auc_matrix = np.zeros((len(domains), len(domains)))
for i, domain1 in enumerate(domains):
    for j, domain2 in enumerate(domains):
        if domain1 == domain2:
            auc_matrix[i, j] = 0
        else:
            auc_matrix[i, j] = np.mean([auc for d, auc in domain_aucs[domain1] if d == domain2])

auc_matrix_df = pd.DataFrame(auc_matrix, columns=domains, index=domains)

############### PLOTTING #########
# Calculate the average value for each row
row_means = np.mean(auc_matrix, axis=1)

# Get the sorted indices of the rows based on the average values in descending order
sorted_indices = np.argsort(-row_means)

# Reorder the rows and columns of the auc_matrix based on the sorted indices
sorted_auc_matrix = auc_matrix[sorted_indices, :][:, sorted_indices]

for i in range(sorted_auc_matrix.shape[0]):
    for j in range(sorted_auc_matrix.shape[1]):
        if i==j:
            sorted_auc_matrix[i,j] = np.nan


# Reorder the domain labels accordingly
sorted_domains = domains[sorted_indices]

# Now you can visualize the sorted_auc_matrix
plt.figure(figsize=(5, 4))
sns.heatmap(sorted_auc_matrix, xticklabels=sorted_domains, yticklabels=sorted_domains, cmap='coolwarm', cbar_kws={'label': 'Distance Measure (AUROC)'})
plt.title('Domain Pairwise Classification Distance')
plt.savefig(f"{save_root}/Pairwise_Classification_SORTED_COMPLETE.png", bbox_inches='tight', dpi=600)
plt.savefig(f"{save_root}/Pairwise_Classification_SORTED_COMPLETE.pdf", bbox_inches='tight', dpi=600)

# plot complete aux_matrix, not sorted
plt.figure(figsize=(5, 4))
for i in range(auc_matrix.shape[0]):
    for j in range(auc_matrix.shape[1]):
        if i==j:
            auc_matrix[i,j] = np.nan
sns.heatmap(auc_matrix, xticklabels=domains, yticklabels=domains, cmap='coolwarm', cbar_kws={'label': 'AUROC (Distance)'})
plt.title('Domain Pairwise Classification Distance')
plt.savefig(f"{save_root}/Pairwise_Classification_NOTSORTED_COMPLETE.png", bbox_inches='tight', dpi=600)
plt.savefig(f"{save_root}/Pairwise_Classification_NOTSORTED_COMPLETE.pdf", bbox_inches='tight', dpi=600)


# plot half sorted_auc_matrix
plt.figure(figsize=(5, 4))
for i in range(sorted_auc_matrix.shape[0]):
    for j in range(sorted_auc_matrix.shape[1]):
        if i<j:
            sorted_auc_matrix[i,j] = np.nan
sns.heatmap(sorted_auc_matrix[1:, :-1], xticklabels=sorted_domains[:-1], yticklabels=sorted_domains[1:], cmap='coolwarm', cbar_kws={'label': 'AUROC (Distance)'})
plt.title('Domain Pairwise Classification Distance')
plt.savefig(f"{save_root}/Pairwise_Classification_SORTED_HALF.png", bbox_inches='tight', dpi=600)
plt.savefig(f"{save_root}/Pairwise_Classification_SORTED_HALF.pdf", bbox_inches='tight', dpi=600)