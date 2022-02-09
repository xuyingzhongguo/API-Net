from pathlib import Path
import pandas as pd
import sys
# import pickle5 as pickle

dfdc_test_path = '/cluster/home/xuyi/xuyi/DFDC/DFDC_validation_single'
dfdc_train_path = '/cluster/home/xuyi/xuyi/DFDC/DFDC_train_single'

dfdc_test_txt_name = 'data_list/dfdc_test.txt'
dfdc_val_txt_name = 'data_list/dfdc_val.txt'
dfdc_train_txt_name = 'data_list/dfdc_train.txt'


def create_list(df_source, df_label, file_name):
    for index, row in df_source.iterrows():
        # label = df_label.where(str(row.path).split('/')[-2] == str(df_label['path']).split('/')[-1][:-4])
        label = df_label.loc[df_label['path'].str.contains(str(row.path).split('/')[-2])]
        # print(label)
        if label.iloc[0, 0]:
            label_ = '1'
        else:
            label_ = '0'

        content = row.path + ' ' + label_ + '\n'
        with open(file_name, 'a+') as ff:
            ff.write(content)
    return 0


def main():
    # test
    # dfdc_val_list = pd.Series(Path(dfdc_test_path).rglob('*.png'))
    # df_source = pd.DataFrame(dfdc_val_list.sort_values(), columns=['path'])
    # df_source['path'] = df_source['path'].astype(str)
    #
    # for index, row in df_source.iterrows():
    #     label = int(row.path.split('/')[7])
    #
    #     if label == 0:
    #         # df_source.loc[index, 'use'] = 'test'
    #         content = row.path + ' ' + str(label) + '\n'
    #         with open(dfdc_test_txt_name, 'a+') as ff:
    #             ff.write(content)
    #     elif label == 1:
    #         # df_source.loc[index, 'use'] = 'val'
    #         content = row.path + ' ' + str(label) + '\n'
    #         with open(dfdc_tset_txt_name, 'a+') as ff:
    #             ff.write(content)
    #     else:
    #         sys.exit('wrong label')


    # train and validation

    # with open('/cluster/home/xuyi/xuyi/DFDC/dfdc_videos.pkl', "rb") as fh:
    #     df_label = pickle.load(fh)
    df_label = pd.read_pickle('/cluster/home/xuyi/xuyi/DFDC/dfdc_videos.pkl')

    dfdc_train_list = pd.Series(Path(dfdc_train_path).rglob('*.png'))
    df_source = pd.DataFrame(dfdc_train_list.sort_values(), columns=['path'])
    df_source['path'] = df_source['path'].astype(str)

    train_num = int(len(df_source) * 0.8)

    # df_train = df_source.loc[:train_num, :]
    # df_val = df_source.loc[train_num:, :]

    df_train = df_source.iloc[:train_num, :]
    df_val = df_source.iloc[train_num:, :]

    create_list(df_val, df_label, dfdc_val_txt_name)
    create_list(df_train, df_label, dfdc_train_txt_name)



if __name__ == "__main__":
    main()
