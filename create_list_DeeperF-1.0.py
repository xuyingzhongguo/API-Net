from pathlib import Path
import pandas as pd
import sys

deeperf_manipulated_test_path = '/cluster/home/xuyi/xuyi/DeeperForensics-1.0/code/DeeperF-1.0_faces_single/manipulated/test'
deeperf_manipulated_val_path = '/cluster/home/xuyi/xuyi/DeeperForensics-1.0/code/DeeperF-1.0_faces_single/manipulated/val'
deeperf_manipulated_train_path = '/cluster/home/xuyi/xuyi/DeeperForensics-1.0/code/DeeperF-1.0_faces_single/manipulated/train'

deeperf_source_test_path = '/cluster/home/xuyi/xuyi/DeeperForensics-1.0/code/DeeperF-1.0_faces_single/source/test'
deeperf_source_val_path = '/cluster/home/xuyi/xuyi/DeeperForensics-1.0/code/DeeperF-1.0_faces_single/source/val'
deeperf_source_train_path = '/cluster/home/xuyi/xuyi/DeeperForensics-1.0/code/DeeperF-1.0_faces_single/source/train'

deeperf_train_txt_name = 'data_list/deeperf_train.txt'
deeperf_val_txt_name = 'data_list/deeperf_val.txt'
deeperf_test_txt_name = 'data_list/deeperf_test.txt'


def func(path):
    deeperf_list = pd.Series(Path(path).rglob('*.png'))
    df_source = pd.DataFrame(deeperf_list.sort_values(), columns=['path'])
    df_source['path'] = df_source['path'].astype(str)

    for index, row in df_source.iterrows():
        use = row.path.split('/')[9]
        if row.path.split('/')[8] == 'source':
            label = '0'
        elif row.path.split('/')[8] == 'manipulated':
            label = '1'
        else:
            sys.exit('wrong row.path.split[8]')

        if use == 'test':
            content = row.path + ' ' + label + '\n'
            with open(deeperf_test_txt_name, 'a+') as ff:
                ff.write(content)
        elif use == 'val':
            content = row.path + ' ' + label + '\n'
            with open(deeperf_val_txt_name, 'a+') as ff:
                ff.write(content)
        elif use == 'train':
            content = row.path + ' ' + label + '\n'
            with open(deeperf_train_txt_name, 'a+') as ff:
                ff.write(content)
        else:
            sys.exit('wrong use')


def main():
    func(deeperf_manipulated_test_path)
    func(deeperf_manipulated_val_path)
    func(deeperf_manipulated_train_path)
    func(deeperf_source_test_path)
    func(deeperf_source_val_path)
    func(deeperf_source_train_path)


if __name__ == "__main__":
    main()
