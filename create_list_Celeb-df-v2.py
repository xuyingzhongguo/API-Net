from pathlib import Path
import pandas as pd
import sys

celeb_real_path = '/cluster/home/xuyi/xuyi/Celeb-DF/code/Celeb-DF-v2_faces_single/Celeb-real'
celeb_synthesis_path = '/cluster/home/xuyi/xuyi/Celeb-DF/code/Celeb-DF-v2_faces_single/Celeb-synthesis'
youtube_real_path = '/cluster/home/xuyi/xuyi/Celeb-DF/code/Celeb-DF-v2_faces_single/YouTube-real'

celeb_df_train_txt_name = 'celeb_df_train.txt'
celeb_df_val_txt_name = 'celeb_df_val.txt'
celeb_df_test_txt_name = 'celeb_df_test.txt'


def main():
    # Celeb_real
    celeb_real_list = pd.Series(Path(celeb_real_path).rglob('*.png'))
    df_source = pd.DataFrame(celeb_real_list.sort_values(), columns=['path'])
    df_source['path'] = df_source['path'].astype(str)
    df_source['label'] = 0
    for index, row in df_source.iterrows():
        source_id = int(row.path.split('/')[-2].split('_')[0].replace('id', ''))

        if source_id > 50:
            # df_source.loc[index, 'use'] = 'test'
            content = row.path + ' ' + str(row.label) + '\n'
            with open(celeb_df_test_txt_name, 'a+') as ff:
                ff.write(content)
        elif 39 < source_id < 51:
            # df_source.loc[index, 'use'] = 'val'
            content = row.path + ' ' + str(row.label) + '\n'
            with open(celeb_df_val_txt_name, 'a+') as ff:
                ff.write(content)
        else:
            # df_source.loc[index, 'use'] = 'train'
            content = row.path + ' ' + str(row.label) + '\n'
            with open(celeb_df_train_txt_name, 'a+') as ff:
                ff.write(content)

    # # Celeb_synthesis
    celeb_synthesis_list = pd.Series(Path(celeb_synthesis_path).rglob('*.png'))
    df_source = pd.DataFrame(celeb_synthesis_list.sort_values(), columns=['path'])
    df_source['path'] = df_source['path'].astype(str)
    df_source['label'] = 1
    for index, row in df_source.iterrows():
        source_id = int(row.path.split('/')[-2].split('_')[0].replace('id', ''))

        if source_id > 50:
            # df_source.loc[index, 'use'] = 'test'
            content = row.path + ' ' + str(row.label) + '\n'
            with open(celeb_df_test_txt_name, 'a+') as ff:
                ff.write(content)
        elif 39 < source_id < 51:
            # df_source.loc[index, 'use'] = 'val'
            content = row.path + ' ' + str(row.label) + '\n'
            with open(celeb_df_val_txt_name, 'a+') as ff:
                ff.write(content)
        else:
            # df_source.loc[index, 'use'] = 'train'
            content = row.path + ' ' + str(row.label) + '\n'
            with open(celeb_df_train_txt_name, 'a+') as ff:
                ff.write(content)

    # youtube_real
    youtube_real_list = pd.Series(Path(youtube_real_path).rglob('*.png'))
    df_source = pd.DataFrame(youtube_real_list.sort_values(), columns=['path'])
    df_source['path'] = df_source['path'].astype(str)
    df_source['label'] = 0
    for index, row in df_source.iterrows():
        source_id = int(row.path.split('/')[-2])

        if source_id > 239:
            # df_source.loc[index, 'use'] = 'test'
            content = row.path + ' ' + str(row.label) + '\n'
            with open(celeb_df_test_txt_name, 'a+') as ff:
                ff.write(content)
        elif 179 < source_id < 240:
            # df_source.loc[index, 'use'] = 'val'
            content = row.path + ' ' + str(row.label) + '\n'
            with open(celeb_df_val_txt_name, 'a+') as ff:
                ff.write(content)
        else:
            # df_source.loc[index, 'use'] = 'train'
            content = row.path + ' ' + str(row.label) + '\n'
            with open(celeb_df_train_txt_name, 'a+') as ff:
                ff.write(content)


if __name__ == "__main__":
    main()
