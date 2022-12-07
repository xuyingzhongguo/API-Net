import pandas as pd
from scipy.special import softmax
import sys
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
import numpy as np

# input_detail_file = '/cluster/home/xuyi/xuyi/API-Net/test_results/3556841/FakeAV/FakeAV_random500_detail.txt'
# output_score_name = '/cluster/home/xuyi/xuyi/API-Net/test_results/3556841/FakeAV/FakeAV_video_wise_score.txt'

# input_detail_file = '/cluster/home/xuyi/xuyi/API-Net/test_results/3556841/DFD/DFD_test_detail.txt'
# output_score_name = '/cluster/home/xuyi/xuyi/API-Net/test_results/3556841/DFD/DFD_video_wise_score.txt'

# input_detail_file = '/cluster/home/xuyi/xuyi/API-Net/test_results/3556841/KoDF/KoDF_test_detail.txt'
# output_score_name = '/cluster/home/xuyi/xuyi/API-Net/test_results/3556841/KoDF/KoDF_video_wise_score.txt'

# input_detail_file = '/cluster/home/xuyi/xuyi/API-Net/test_results/3556841/Celeb/Celeb_test_detail.txt'
# output_score_name = '/cluster/home/xuyi/xuyi/API-Net/test_results/3556841/Celeb/Celeb_video_wise_score.txt'

# input_detail_file = '/cluster/home/xuyi/xuyi/API-Net/test_results/3556841/DFDC/DFDC_test_detail.txt'
# output_score_name = '/cluster/home/xuyi/xuyi/API-Net/test_results/3556841/DFDC/DFDC_video_wise_score.txt'

# input_detail_file = '/cluster/home/xuyi/xuyi/API-Net/test_results/3444101/close/df_test_detail.txt'
# output_score_name = '/cluster/home/xuyi/xuyi/API-Net/test_results/3444101/close/df_video_wise_score.txt'

# input_detail_file = '/cluster/home/xuyi/xuyi/API-Net/test_results/3543488/close/f2f_test_detail.txt'
# output_score_name = '/cluster/home/xuyi/xuyi/API-Net/test_results/3543488/close/f2f_video_wise_score.txt'

# input_detail_file = '/cluster/home/xuyi/xuyi/API-Net/test_results/3543488/close/fs_test_detail.txt'
# output_score_name = '/cluster/home/xuyi/xuyi/API-Net/test_results/3543488/close/fs_video_wise_score.txt'

# input_detail_file = '/cluster/home/xuyi/xuyi/API-Net/test_results/3543488/close/nt_test_detail.txt'
# output_score_name = '/cluster/home/xuyi/xuyi/API-Net/test_results/3543488/close/nt_video_wise_score.txt'

oc = 'close'
job_id = '3209785'
jobid = '3220079_model'
# input_detail_file = f"test_results/{job_id}/close/df_test_detail.txt"
# output_score_name = f"test_results/{job_id}/close/df_video_wise_score.txt"
# input_detail_file = f"test_results/{job_id}/{oc}/f2f_test_detail.txt"
# output_score_name = f"test_results/{job_id}/{oc}/f2f_video_wise_score.txt"
# input_detail_file = f"test_results/{job_id}/{oc}/fs_test_detail.txt"
# output_score_name = f"test_results/{job_id}/{oc}/fs_video_wise_score.txt"
input_detail_file = f"test_results/{job_id}/{oc}/nt_test_detail.txt"
output_score_name = f"test_results/{job_id}/{oc}/nt_video_wise_score.txt"

# input_detail_file = f"test_results/{job_id}/FakeAV/FakeAV_random500_detail.txt"
# output_score_name = f"test_results/{job_id}/FakeAV/FakeAV_video_wise_score.txt"
# input_detail_file = f"test_results/{job_id}/KoDF/KoDF_test_detail.txt"
# output_score_name = f"test_results/{job_id}/KoDF/KoDF_video_wise_score.txt"
# input_detail_file = f"test_results/{job_id}/DFD/DFD_test_detail.txt"
# output_score_name = f"test_results/{job_id}/DFD/DFD_video_wise_score.txt"
# input_detail_file = f"test_results/{job_id}/Celeb/Celeb_test_detail.txt"
# output_score_name = f"test_results/{job_id}/Celeb/Celeb_video_wise_score.txt"
# input_detail_file = f"test_results/{job_id}/DFDC/DFDC_test_detail.txt"
# output_score_name = f"test_results/{job_id}/DFDC/DFDC_video_wise_score.txt"


def pd_at_far(y_true, y_score, tnr_th=0.9):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return np.interp(1.0-tnr_th, fpr, tpr)


def pd_auc_metrics(y_true, y_score, tnr_th=0.9, acc_th=0.5):
    auc_score = roc_auc_score(y_true, y_score)
    pd = pd_at_far(y_true, y_score)
    y_score = np.asarray(y_score)
    y_pred = np.asarray(y_score>acc_th, dtype=np.uint8)
    acc = balanced_accuracy_score(y_true, y_pred)
    return acc*100., pd*100., auc_score*100.


pd.set_option('display.max_columns', None)
df = pd.read_csv(input_detail_file, delimiter='\t', names=['path', 's0', 's1', 's2', 's3', 's4', 'label', 'prediction'])
df['path_folder'] = df['path'].apply(lambda x: x.replace(x.split('/')[-1], ''))
df[['s0', 's1', 's2', 's3', 's4', 'label', 'prediction']] = df[['s0', 's1', 's2', 's3', 's4', 'label', 'prediction']].apply(pd.to_numeric)
df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
df_score = df[['s0', 's1', 's2', 's3', 's4']]

df_softmax = softmax(df_score, axis=1)
df_softmax['s1234'] = df_softmax[['s1', 's2', 's3', 's4']].sum(axis=1)
df['s0'] = df_softmax['s0']
df['s1'] = df_softmax['s1234']
df.drop(columns=['s2', 's3', 's4', 'path', 'prediction'], inplace=True)
# df_final = df.groupby(['path_folder']).mean()
df_final = df

df_final['prediction'] = df_final['s0'].apply(lambda x: 1 if x < 0.5 else 0)
df_final.to_csv(output_score_name)
tp = tn = fp = fn = 0
for index, row in df_final.iterrows():
    if row['label'] == 1 and row['prediction'] == 1:
        tp = tp + 1
    elif row['label'] == 1 and row['prediction'] == 0:
        fn = fn + 1
    elif row['label'] == 0 and row['prediction'] == 1:
        fp = fp + 1
    elif row['label'] == 0 and row['prediction'] == 0:
        tn = tn + 1
    else:
        print('label and prediction wrong value')
        sys.exit()
print(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}')

acc, pd, auc = pd_auc_metrics(df_final['label'], df_final['s1'], acc_th=0.5)

print(input_detail_file)
print("Video Wise")
print("Accuracy:", acc)
print("AUC:", auc)
print("pd@10:", pd)
