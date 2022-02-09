from pathlib import Path
import pandas as pd
import sys

faceshifter_val_path = '/cluster/home/xuyi/xuyi/FaceForensics++/manipulated_sequences/FaceShifter'

faceshifter_val_txt_name = 'data_list/faceshifter_test.txt'


def main():
    faceshifter_val_list = pd.Series(Path(faceshifter_val_path).rglob('*.png'))
    df_source = pd.DataFrame(faceshifter_val_list.sort_values(), columns=['path'])
    df_source['path'] = df_source['path'].astype(str)

    for index, row in df_source.iterrows():
        id = row.path.split('/')[-2].split('_')[0]
        if int(id) > 799:
            content = row.path + ' ' + '1' + '\n'
            with open(faceshifter_val_txt_name, 'a+') as ff:
                ff.write(content)


if __name__ == "__main__":
    main()
