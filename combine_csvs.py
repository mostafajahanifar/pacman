import pandas as pd

from survival_utils import combine_csvs, combine_feats_multipe_WSI

##script for combining features from multiple csv files into one csv file

def discovery_validation():
    path_to_clinical_csv = '/data/data/NOTT/cell_surv/set1to18_5folds/3fold_split_03052022/folds_exp/brace_all_disc_val.csv'
    path_to_save_combined_csv = '/data/data/NOTT/cell_surv/set1to18_5folds/exps/HGC_paper_related/all_feats_combi/features_combined/NOTT/discovery_valid_combined.csv'
    
    path_to_grades_feats = '/data/data/NOTT/cell_surv/set1to18_5folds/exps/HGC_paper_related/all_feats_combi/features/NOTT/grade_feats.csv'
    path_to_regionpatch_feats = '/data/data/NOTT/cell_surv/set1to18_5folds/exps/HGC_paper_related/all_feats_combi/features/NOTT/regionpatch_feats.csv'
    path_to_ROI_mitosis_feats = '/data/data/NOTT/cell_surv/set1to18_5folds/exps/HGC_paper_related/all_feats_combi/features/NOTT/ROI_mitosis_feats.csv'
    path_to_TILs_feats = '/data/data/NOTT/cell_surv/set1to18_5folds/exps/HGC_paper_related/all_feats_combi/features/NOTT/TILs_feats.csv'
    
    csv_list = [path_to_grades_feats, path_to_regionpatch_feats, path_to_ROI_mitosis_feats, path_to_TILs_feats]
    #csv_list = [path_to_TILs_feats]

    ##first combine features for multipe WSIs per case for example in Challenging set. This needs to be done once for all the feats csv files
    #for c in csv_list:
    #    combine_feats_multipe_WSI(c)
    #print('Done merging features for multiple WSIs per case')
    #exit()
    
    ##list of cases to exclude. For example cases dropped during image QC.
    skip_list = ['81RMTR','81RLPT','81RNKM','81RRQR','41CHGC','41DCGD','41DCED','41DCEE','41DJJG','81OLSQ','81OOTP','81OPPK','12DBAY','12VVUYC','12VVUZD','12VVVVU','12VVAUB','12VVCVD'] ##all cases. set 1 to 52
    
    print('combined file will be saved at:', path_to_save_combined_csv)
        
    combine_csvs(csv_list, path_to_clinical_csv, path_to_save_combined_csv, skip_list)

##the clincal features (excluding the time/events for test) are provided in a single csv file where as the test set IDs are provided seperately. 
## This function will combine the clinical features according to the test IDs
def add_clinical_feats_to_test_IDs(path_to_test_ids, path_to_clinical_csv_all, path_to_clinical_csv):
    
    clinic_ids = pd.read_csv(path_to_test_ids)
    clinic_ids = clinic_ids['Case ID'].str.split('_', expand=True)[0]
    clinic_ids = pd.DataFrame(clinic_ids.to_list(), columns=['Case ID'])
    clinic_feats = pd.read_csv(path_to_clinical_csv_all)

    clinic_test = clinic_feats.merge(clinic_ids, how='inner', on='Case ID')

    clinic_test.to_csv(path_to_clinical_csv, index=False)

def test():
    path_to_test_ids = '/data/data/NOTT/cell_surv/set1to18_5folds/3fold_split_03052022/Shan_3folds/test/finaltest.csv'
    path_to_clinical_csv_all = '/data/data/NOTT/cell_surv/set1to18_5folds/3fold_split_03052022/Shan_3folds/BRACE_all.csv'
    path_to_clinical_csv = '/data/data/NOTT/cell_surv/set1to18_5folds/3fold_split_03052022/Shan_3folds/BRACE_test.csv'
    path_to_TILs_feats = '/data/data/NOTT/cell_surv/set1to18_5folds/exps/HGC_paper_related/all_feats_combi/features/NOTT/TILs_feats.csv'
    
    add_clinical_feats_to_test_IDs(path_to_test_ids, path_to_clinical_csv_all, path_to_clinical_csv)
    
    path_to_save_combined_csv = '/data/data/NOTT/cell_surv/set1to18_5folds/exps/HGC_paper_related/all_feats_combi/features_combined/NOTT/test_combined.csv'
    
    path_to_grades_feats = '/data/data/NOTT/cell_surv/set1to18_5folds/exps/HGC_paper_related/all_feats_combi/features/NOTT/grade_feats.csv'
    path_to_regionpatch_feats = '/data/data/NOTT/cell_surv/set1to18_5folds/exps/HGC_paper_related/all_feats_combi/features/NOTT/regionpatch_feats.csv'
    path_to_ROI_mitosis_feats = '/data/data/NOTT/cell_surv/set1to18_5folds/exps/HGC_paper_related/all_feats_combi/features/NOTT/ROI_mitosis_feats.csv'
    
    csv_list = [path_to_grades_feats, path_to_regionpatch_feats, path_to_ROI_mitosis_feats, path_to_TILs_feats]
    
    ##first combine features for multiple WSIs per case for example in Challenging set. This needs to be done once for all the feats csv files
    #for c in csv_list:
    #    combine_feats_multipe_WSI(c)
    #print('Done merging features for multiple WSIs per case')
    
    ##list of cases to exclude. For example cases dropped during image QC.
    skip_list = ['81RMTR','81RLPT','81RNKM','81RRQR','41CHGC','41DCGD','41DCED','41DCEE','41DJJG','81OLSQ','81OOTP','81OPPK','12DBAY','12VVUYC','12VVUZD','12VVVVU','12VVAUB','12VVCVD'] ##all cases. set 1 to 52
    
    print('combined file will be saved at:', path_to_save_combined_csv)
        
    combine_csvs(csv_list, path_to_clinical_csv, path_to_save_combined_csv, skip_list)
    print('Done combining csv from multipe feature sets into one csv')

if __name__ == '__main__':
    #discovery_validation()
    test()

