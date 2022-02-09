from pathlib import Path
import pandas as pd
import sys

dfd_path = '/cluster/home/xuyi/xuyi/FaceForensics++/manipulated_sequences/DeepFakeDetection'
original_path = '/cluster/home/xuyi/xuyi/FaceForensics++/original_sequences/actors/raw/face_images/'

dfd_test_txt_name = 'data_list/dfd_test.txt'
dfd_val_txt_name = 'data_list/dfd_val.txt'
dfd_train_txt_name = 'data_list/dfd_train.txt'


def main():
    dfd_list = pd.Series(Path(dfd_path).rglob('*.png'))
    df_source = pd.DataFrame(dfd_list.sort_values(), columns=['path'])
    df_source['path'] = df_source['path'].astype(str)

    for index, row in df_source.iterrows():
        id = row.path.split('/')[-2].split('_')[0]
        if int(id) > 23:
            content = row.path + ' ' + '1' + '\n'
            with open(dfd_test_txt_name, 'a+') as ff:
                ff.write(content)
        elif 18 < int(id) < 24:
            content = row.path + ' ' + '1' + '\n'
            with open(dfd_val_txt_name, 'a+') as ff:
                ff.write(content)
        elif int(id) < 19:
            content = row.path + ' ' + '1' + '\n'
            with open(dfd_train_txt_name, 'a+') as ff:
                ff.write(content)
        else:
            print(row.path)
            sys.exit('wrong path')

    ori_val_list = pd.Series(Path(original_path).rglob('*.png'))
    or_source = pd.DataFrame(ori_val_list.sort_values(), columns=['path'])
    or_source['path'] = or_source['path'].astype(str)

    for index, row in or_source.iterrows():
        id = row.path.split('/')[-2].split('_')[0]
        if int(id) > 23:
            content = row.path + ' ' + '0' + '\n'
            with open(dfd_test_txt_name, 'a+') as ff:
                ff.write(content)
        elif 18 < int(id) < 24:
            content = row.path + ' ' + '0' + '\n'
            with open(dfd_val_txt_name, 'a+') as ff:
                ff.write(content)
        elif int(id) < 19:
            content = row.path + ' ' + '0' + '\n'
            with open(dfd_train_txt_name, 'a+') as ff:
                ff.write(content)
        else:
            print(row.path)
            sys.exit('wrong path')


if __name__ == "__main__":
    main()
