from pathlib import Path
import numpy as np
import random
import argparse
import os
import sys

# how to run:
# python create_list.py -df true -f2f true -nt true --binary_label --name_train data_list/DF_F2F_NT_train_000599_binary_label.txt --name_val data_list/DF_F2F_NT_val_600799_binary_label.txt --name_test data_list/DF_F2F_NT_test_800999_binary_label.txt

face_images_paths = {
    # '': '/home/user1/xuyi/FaceForensics++/DeepFakeDetection/c23/face_images/',
    'deepfakes': '/cluster/home/xuyi/xuyi/FaceForensics++/manipulated_sequences/Deepfakes/raw/face_images/',
    'face2face': '/cluster/home/xuyi/xuyi/FaceForensics++/manipulated_sequences/Face2Face/raw/face_images/',
    'faceswap': '/cluster/home/xuyi/xuyi/FaceForensics++/manipulated_sequences/FaceSwap/raw/face_images/',
    'neuraltexures': '/cluster/home/xuyi/xuyi/FaceForensics++/manipulated_sequences/NeuralTextures/raw/face_images/',
    'original': '/cluster/home/xuyi/xuyi/FaceForensics++/original_sequences/youtube/raw/face_images/',
    'faceshifter': '/cluster/home/xuyi/xuyi/FaceForensics++/manipulated_sequences/FaceShifter/raw/face_images/',
}


def dataset_decide(deepfakes_inc, face2face_inc, faceswap_inc, neuraltextures_inc, original_inc, faceshifter_inc):
    dataset_include = []

    if deepfakes_inc == 'true':
        dataset_include.append(face_images_paths['deepfakes'])
    if face2face_inc == 'true':
        dataset_include.append(face_images_paths['face2face'])
    if faceswap_inc == 'true':
        dataset_include.append(face_images_paths['faceswap'])
    if neuraltextures_inc == 'true':
        dataset_include.append(face_images_paths['neuraltexures'])
    if original_inc == 'true':
        dataset_include.append(face_images_paths['original'])
    if faceshifter_inc == 'true':
        dataset_include.append(face_images_paths['faceshifter'])

    return dataset_include


def find_path(file_paths, train_list_name, val_list_name, test_list_name, binary_label):
    file_name_lists = []
    for file_path in file_paths:
        file_path = Path(file_path)
        file_name_list = list(file_path.glob('**/*.png'))
        file_name_lists += file_name_list

    image_label = np.array(file_name_lists).reshape(-1, 1)
    label = np.ones((len(file_name_lists), 1), dtype=np.int8)
    images_labels = np.hstack((image_label, label))

    for single in images_labels:
        single_path = str(single[0].absolute())
        if binary_label:
            if 'youtube' in single_path.split('/'):
                single_label = '0'
            else:
                single_label = '1'
        else:
            if 'youtube' in single_path.split('/'):
                single_label = '0'
            elif 'Deepfakes' in single_path.split('/'):
                single_label = '1'
            elif 'Face2Face' in single_path.split('/'):
                single_label = '2'
            elif 'FaceSwap' in single_path.split('/'):
                single_label = '3'
            elif 'NeuralTextures' in single_path.split('/'):
                single_label = '3'
            else:
                sys.exit('wrong dataset name')

        content = single_path + ' ' + single_label + '\n'

        if 'youtube' in single_path.split('/'):
            if int(single_path.split('/')[-2]) < 600:
                with open(train_list_name, 'a+') as ff:
                    ff.write(content)
            elif 599 < int(single_path.split('/')[-2]) < 800:
                with open(val_list_name, 'a+') as ff:
                    ff.write(content)
            elif int(single_path.split('/')[-2]) > 799:
                with open(test_list_name, 'a+') as ff:
                    ff.write(content)
            else:
                sys.exit('wrong path')
        else:
            if int(single_path.split('/')[-2][0:3]) < 600:
                with open(train_list_name, 'a+') as ff:
                    ff.write(content)
            elif 599 < int(single_path.split('/')[-2][0:3]) < 800:
                with open(val_list_name, 'a+') as ff:
                    ff.write(content)
            elif int(single_path.split('/')[-2][0:3]) > 799:
                with open(test_list_name, 'a+') as ff:
                    ff.write(content)
            else:
                sys.exit('wrong path')



def main():
    args = parse.parse_args()
    deepfakes_inc = args.deepfakes
    face2face_inc = args.face2face
    faceswap_inc = args.faceswap
    neuraltextures_inc = args.neuraltextures
    original_inc = args.original
    faceshifter_inc = args.faceshifter
    train_list_name = args.name_train
    val_list_name = args.name_val
    test_list_name = args.name_test
    binary_label = args.binary_label

    dataset_include = dataset_decide(deepfakes_inc, face2face_inc, faceswap_inc, neuraltextures_inc, original_inc,
                                     faceshifter_inc)
    find_path(dataset_include, train_list_name, val_list_name, test_list_name, binary_label)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--deepfakes', '-df', type=str, default='false')
    parse.add_argument('--face2face', '-f2f', type=str, default='false')
    parse.add_argument('--faceswap', '-fs', type=str, default='false')
    parse.add_argument('--neuraltextures', '-nt', type=str, default='false')
    parse.add_argument('--original', '-o', type=str, default='true')
    parse.add_argument('--faceshifter', '-fsh', type=str, default='false')
    parse.add_argument('--name_train', type=str, default='train.txt')
    parse.add_argument('--name_val', type=str, default='val.txt')
    parse.add_argument('--name_test', type=str, default='test.txt')
    parse.add_argument('--binary_label', action='store_true')

    main()
