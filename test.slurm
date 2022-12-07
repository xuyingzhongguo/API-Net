#!/bin/bash
#SBATCH --job-name="API-test"   # Sensible name for the job
#SBATCH --account=ie-idi   # Account for consumed resources
#SBATCH --nodes=1           # Allocate 1 nodes for the job
#SBATCH -c 8                 # Number of cores (can vary)
#SBATCH --gres=gpu:A100m80
#SBATCH --time=00-02:30:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --partition=GPUQ
#SBATCH --mail-user=ying.xu@ntnu.no
#SBATCH --ntasks-per-node=1     # 1 task per node

#SBATCH --priority=TOP
#SBATCH --no-requeue
module purge
cd ${SLURM_SUBMIT_DIR}
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

#module load intel/2020b
module load Python/3.8.6-GCCcore-10.2.0
source A100-venv/bin/activate
#source venv/bin/activate

#echo "test 2475004 model"
#python -u test.py -b 1 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path test_results/df_test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee test_results/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path test_results/f2f_test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee test_results/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path test_results/fs_test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee test_results/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --test_list data_list/neuraltextures_test_800999_frame30.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path test_results/nt_test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee test_results/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2475034 model"
#python -u test.py -b 1 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2475034/model_best.pth.tar --output_path test_results/2475034/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2475034/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2475034/model_best.pth.tar --output_path test_results/2475034/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2475034/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/2475034/model_best.pth.tar --output_path test_results/2475034/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2475034/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --test_list data_list/neuraltextures_test_800999_frame30.txt --resume model_save/res101/2475034/model_best.pth.tar --output_path test_results/2475034/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2475034/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2506357 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/close/fs_test_${SLURM_JOB_ID}.txt

#echo "test 2506357 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/2506357/model_best.pth.tar --output_path test_results/2506357/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506357/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2506358 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2506358 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label4.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2506358/model_best.pth.tar --output_path test_results/2506358/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506358/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2506359 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2506359 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2506359/model_best.pth.tar --output_path test_results/2506359/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506359/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2506360 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label1.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2506360 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label4.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label1.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2506360/model_best.pth.tar --output_path test_results/2506360/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2506360/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2521041 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2521041 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/efficientnet/2521041/model_best.pth.tar --output_path test_results/2521041/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521041/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2521040 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2521040 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/efficientnet/2521040/model_best.pth.tar --output_path test_results/2521040/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2521040/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2532498 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2532498 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb6/2532498/model_best.pth.tar --output_path test_results/2532498/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2532498/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2543837 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2543837 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb7/2543837/model_best.pth.tar --output_path test_results/2543837/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2543837/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2546248 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2546248 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb0/2546248/model_best.pth.tar --output_path test_results/2546248/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546248/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2546254 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2546254 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb1/2546254/model_best.pth.tar --output_path test_results/2546254/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546254/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2546255 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2546255 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb2/2546255/model_best.pth.tar --output_path test_results/2546255/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546255/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2546256 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2546256 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb3/2546256/model_best.pth.tar --output_path test_results/2546256/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546256/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2546261 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2546261 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb4/2546261/model_best.pth.tar --output_path test_results/2546261/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546261/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2546262 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2546262 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb5/2546262/model_best.pth.tar --output_path test_results/2546262/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2546262/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2560369 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb1/2560369/model_best.pth.tar --output_path test_results/2560369/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560369/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb1/2560369/model_best.pth.tar --output_path test_results/2560369/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560369/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb1/2560369/model_best.pth.tar --output_path test_results/2560369/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560369/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2560369 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb1/2560369/model_best.pth.tar --output_path test_results/2560369/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560369/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb1/2560369/model_best.pth.tar --output_path test_results/2560369/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560369/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb1/2560369/model_best.pth.tar --output_path test_results/2560369/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560369/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb1/2560369/model_best.pth.tar --output_path test_results/2560369/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560369/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2560370 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb2/2560370/model_best.pth.tar --output_path test_results/2560370/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560370/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb2/2560370/model_best.pth.tar --output_path test_results/2560370/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560370/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb2/2560370/model_best.pth.tar --output_path test_results/2560370/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560370/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2560370 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb2/2560370/model_best.pth.tar --output_path test_results/2560370/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560370/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb2/2560370/model_best.pth.tar --output_path test_results/2560370/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560370/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb2/2560370/model_best.pth.tar --output_path test_results/2560370/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560370/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb2/2560370/model_best.pth.tar --output_path test_results/2560370/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560370/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2560371 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb3/2560371/model_best.pth.tar --output_path test_results/2560371/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560371/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb3/2560371/model_best.pth.tar --output_path test_results/2560371/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560371/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb3/2560371/model_best.pth.tar --output_path test_results/2560371/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560371/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2560371 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb3/2560371/model_best.pth.tar --output_path test_results/2560371/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560371/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb3/2560371/model_best.pth.tar --output_path test_results/2560371/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560371/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb3/2560371/model_best.pth.tar --output_path test_results/2560371/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560371/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb3/2560371/model_best.pth.tar --output_path test_results/2560371/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2560371/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2568987 model closed set"
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2568987/model_best.pth.tar --output_path test_results/2568987/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568987/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2568987/model_best.pth.tar --output_path test_results/2568987/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568987/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2568987/model_best.pth.tar --output_path test_results/2568987/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568987/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2568987 model open set"
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2568987/model_best.pth.tar --output_path test_results/2568987/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568987/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/2568987/model_best.pth.tar --output_path test_results/2568987/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568987/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2568987/model_best.pth.tar --output_path test_results/2568987/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568987/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2568987/model_best.pth.tar --output_path test_results/2568987/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568987/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2568988 model closed set"
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb0/2568988/model_best.pth.tar --output_path test_results/2568988/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568988/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb0/2568988/model_best.pth.tar --output_path test_results/2568988/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568988/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb0/2568988/model_best.pth.tar --output_path test_results/2568988/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568988/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2568988 model open set"
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb0/2568988/model_best.pth.tar --output_path test_results/2568988/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568988/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb0/2568988/model_best.pth.tar --output_path test_results/2568988/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568988/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb0/2568988/model_best.pth.tar --output_path test_results/2568988/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568988/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb0 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb0/2568988/model_best.pth.tar --output_path test_results/2568988/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568988/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2568989 model closed set"
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb1/2568989/model_best.pth.tar --output_path test_results/2568989/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568989/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb1/2568989/model_best.pth.tar --output_path test_results/2568989/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568989/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb1/2568989/model_best.pth.tar --output_path test_results/2568989/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568989/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2568989 model open set"
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb1/2568989/model_best.pth.tar --output_path test_results/2568989/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568989/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb1/2568989/model_best.pth.tar --output_path test_results/2568989/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568989/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb1/2568989/model_best.pth.tar --output_path test_results/2568989/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568989/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb1 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb1/2568989/model_best.pth.tar --output_path test_results/2568989/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568989/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2568990 model closed set"
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb2/2568990/model_best.pth.tar --output_path test_results/2568990/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568990/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb2/2568990/model_best.pth.tar --output_path test_results/2568990/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568990/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb2/2568990/model_best.pth.tar --output_path test_results/2568990/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568990/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2568990 model open set"
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb2/2568990/model_best.pth.tar --output_path test_results/2568990/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568990/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb2/2568990/model_best.pth.tar --output_path test_results/2568990/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568990/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb2/2568990/model_best.pth.tar --output_path test_results/2568990/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568990/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb2 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb2/2568990/model_best.pth.tar --output_path test_results/2568990/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568990/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2568991 model closed set"
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb3/2568991/model_best.pth.tar --output_path test_results/2568991/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568991/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb3/2568991/model_best.pth.tar --output_path test_results/2568991/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568991/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb3/2568991/model_best.pth.tar --output_path test_results/2568991/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568991/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2568991 model open set"
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb3/2568991/model_best.pth.tar --output_path test_results/2568991/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568991/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb3/2568991/model_best.pth.tar --output_path test_results/2568991/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568991/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb3/2568991/model_best.pth.tar --output_path test_results/2568991/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568991/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb3 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb3/2568991/model_best.pth.tar --output_path test_results/2568991/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568991/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2568992 model closed set"
#python -u test.py -b 1 --dist_type cosine --n_classes_total 4 --model_name effb4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb4/2568992/model_best.pth.tar --output_path test_results/2568992/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568992/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --dist_type cosine --n_classes_total 4 --model_name effb4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb4/2568992/model_best.pth.tar --output_path test_results/2568992/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568992/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --dist_type cosine --n_classes_total 4 --model_name effb4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb4/2568992/model_best.pth.tar --output_path test_results/2568992/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568992/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2568992 model open set"
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb4/2568992/model_best.pth.tar --output_path test_results/2568992/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568992/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb4/2568992/model_best.pth.tar --output_path test_results/2568992/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568992/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb4/2568992/model_best.pth.tar --output_path test_results/2568992/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568992/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb4/2568992/model_best.pth.tar --output_path test_results/2568992/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568992/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2568993 model closed set"
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb5/2568993/model_best.pth.tar --output_path test_results/2568993/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568993/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb5/2568993/model_best.pth.tar --output_path test_results/2568993/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568993/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb5/2568993/model_best.pth.tar --output_path test_results/2568993/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568993/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2568993 model open set"
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb5/2568993/model_best.pth.tar --output_path test_results/2568993/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568993/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb5/2568993/model_best.pth.tar --output_path test_results/2568993/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568993/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb5/2568993/model_best.pth.tar --output_path test_results/2568993/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568993/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb5 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb5/2568993/model_best.pth.tar --output_path test_results/2568993/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568993/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2568994 model closed set"
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb6/2568994/model_best.pth.tar --output_path test_results/2568994/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568994/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb6/2568994/model_best.pth.tar --output_path test_results/2568994/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568994/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb6/2568994/model_best.pth.tar --output_path test_results/2568994/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568994/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2568994 model open set"
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb6/2568994/model_best.pth.tar --output_path test_results/2568994/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568994/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb6/2568994/model_best.pth.tar --output_path test_results/2568994/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568994/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb6/2568994/model_best.pth.tar --output_path test_results/2568994/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568994/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb6 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb6/2568994/model_best.pth.tar --output_path test_results/2568994/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568994/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2568995 model closed set"
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb7/2568995/model_best.pth.tar --output_path test_results/2568995/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568995/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/effb7/2568995/model_best.pth.tar --output_path test_results/2568995/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568995/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb7/2568995/model_best.pth.tar --output_path test_results/2568995/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568995/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2568995 model open set"
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/effb7/2568995/model_best.pth.tar --output_path test_results/2568995/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568995/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/effb7/2568995/model_best.pth.tar --output_path test_results/2568995/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568995/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/effb7/2568995/model_best.pth.tar --output_path test_results/2568995/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568995/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --dist_type cosine -b 1 --n_classes_total 4 --model_name effb7 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/effb7/2568995/model_best.pth.tar --output_path test_results/2568995/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2568995/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2569111 model closed set"
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2569111/model_best.pth.tar --output_path test_results/2569111/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569111/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2569111/model_best.pth.tar --output_path test_results/2569111/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569111/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2569111/model_best.pth.tar --output_path test_results/2569111/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569111/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2569111 model open set"
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2569111/model_best.pth.tar --output_path test_results/2569111/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569111/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/2569111/model_best.pth.tar --output_path test_results/2569111/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569111/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2569111/model_best.pth.tar --output_path test_results/2569111/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569111/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2569111/model_best.pth.tar --output_path test_results/2569111/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569111/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2592377 model closed set"
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2592377/model_best.pth.tar --output_path test_results/2592377/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592377/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2592377/model_best.pth.tar --output_path test_results/2592377/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592377/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2592377/model_best.pth.tar --output_path test_results/2592377/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592377/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2592377 model open set"
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2592377/model_best.pth.tar --output_path test_results/2592377/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592377/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/2592377/model_best.pth.tar --output_path test_results/2592377/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592377/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2592377/model_best.pth.tar --output_path test_results/2592377/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592377/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2592377/model_best.pth.tar --output_path test_results/2592377/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592377/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2592378 model closed set"
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2592378/model_best.pth.tar --output_path test_results/2592378/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592378/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2592378/model_best.pth.tar --output_path test_results/2592378/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592378/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2592378/model_best.pth.tar --output_path test_results/2592378/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592378/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2592378 model open set"
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2592378/model_best.pth.tar --output_path test_results/2592378/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592378/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/2592378/model_best.pth.tar --output_path test_results/2592378/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592378/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2592378/model_best.pth.tar --output_path test_results/2592378/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592378/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2592378/model_best.pth.tar --output_path test_results/2592378/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592378/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2592784 model closed set"
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2592784/model_best.pth.tar --output_path test_results/2592784/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592784/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2592784/model_best.pth.tar --output_path test_results/2592784/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592784/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2592784/model_best.pth.tar --output_path test_results/2592784/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592784/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2592784 model open set"
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2592784/model_best.pth.tar --output_path test_results/2592784/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592784/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/2592784/model_best.pth.tar --output_path test_results/2592784/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592784/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2592784/model_best.pth.tar --output_path test_results/2592784/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592784/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2592784/model_best.pth.tar --output_path test_results/2592784/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592784/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2592785 model closed set"
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2592785/model_best.pth.tar --output_path test_results/2592785/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592785/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2592785/model_best.pth.tar --output_path test_results/2592785/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592785/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2592785/model_best.pth.tar --output_path test_results/2592785/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592785/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2592785 model open set"
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2592785/model_best.pth.tar --output_path test_results/2592785/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592785/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/2592785/model_best.pth.tar --output_path test_results/2592785/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592785/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2592785/model_best.pth.tar --output_path test_results/2592785/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592785/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader bvcb_loader -b 1 --n_classes_total 4 --model_name res101 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2592785/model_best.pth.tar --output_path test_results/2592785/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2592785/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2619307 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2619307/model_best.pth.tar --output_path test_results/2619307/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2619307/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2619307/model_best.pth.tar --output_path test_results/2619307/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2619307/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2619307/model_best.pth.tar --output_path test_results/2619307/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2619307/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2619307 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2619307/model_best.pth.tar --output_path test_results/2619307/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2619307/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/2619307/model_best.pth.tar --output_path test_results/2619307/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2619307/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2619307/model_best.pth.tar --output_path test_results/2619307/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2619307/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2619307/model_best.pth.tar --output_path test_results/2619307/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2619307/open/nt_test_${SLURM_JOB_ID}.txt



#echo "test 2788474 model closed set"
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2788474/model_best.pth.tar --output_path test_results/2788474/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788474/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2788474/model_best.pth.tar --output_path test_results/2788474/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788474/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/2788474/model_best.pth.tar --output_path test_results/2788474/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788474/close/fs_test_${SLURM_JOB_ID}.txt

#echo "test 2788474 model open set"
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2788474/model_best.pth.tar --output_path test_results/2788474/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788474/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2788474/model_best.pth.tar --output_path test_results/2788474/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788474/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/2788474/model_best.pth.tar --output_path test_results/2788474/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788474/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/2788474/model_best.pth.tar --output_path test_results/2788474/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788474/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2788475 model closed set"
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2788475/model_best.pth.tar --output_path test_results/2788475/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788475/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2788475/model_best.pth.tar --output_path test_results/2788475/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788475/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2788475/model_best.pth.tar --output_path test_results/2788475/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788475/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2788475 model open set"
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2788475/model_best.pth.tar --output_path test_results/2788475/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788475/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2788475/model_best.pth.tar --output_path test_results/2788475/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788475/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label4.txt --resume model_save/res101/2788475/model_best.pth.tar --output_path test_results/2788475/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788475/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2788475/model_best.pth.tar --output_path test_results/2788475/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788475/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2788477 model closed set"
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2788477/model_best.pth.tar --output_path test_results/2788477/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788477/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2788477/model_best.pth.tar --output_path test_results/2788477/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788477/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2788477/model_best.pth.tar --output_path test_results/2788477/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788477/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2788477 model open set"
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2788477/model_best.pth.tar --output_path test_results/2788477/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788477/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2788477/model_best.pth.tar --output_path test_results/2788477/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788477/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label4.txt --resume model_save/res101/2788477/model_best.pth.tar --output_path test_results/2788477/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788477/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2788477/model_best.pth.tar --output_path test_results/2788477/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788477/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2788478 model closed set"
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label1.txt --resume model_save/res101/2788478/model_best.pth.tar --output_path test_results/2788478/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788478/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2788478/model_best.pth.tar --output_path test_results/2788478/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788478/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2788478/model_best.pth.tar --output_path test_results/2788478/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788478/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2788478 model open set"
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label4.txt --resume model_save/res101/2788478/model_best.pth.tar --output_path test_results/2788478/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788478/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label1.txt --resume model_save/res101/2788478/model_best.pth.tar --output_path test_results/2788478/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788478/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2788478/model_best.pth.tar --output_path test_results/2788478/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788478/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2788478/model_best.pth.tar --output_path test_results/2788478/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788478/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2788473 model open set"
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2788473/model_best.pth.tar --output_path test_results/2788473/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788473/open/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2788473/model_best.pth.tar --output_path test_results/2788473/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788473/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/2788473/model_best.pth.tar --output_path test_results/2788473/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788473/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/2788473/model_best.pth.tar --output_path test_results/2788473/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2788473/open/nt_test_${SLURM_JOB_ID}.txt




#echo "test 2867649 model closed set"
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/2867649/model_best.pth.tar --output_path test_results/2867649/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867649/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label2.txt --resume model_save/res101/2867649/model_best.pth.tar --output_path test_results/2867649/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867649/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label3.txt --resume model_save/res101/2867649/model_best.pth.tar --output_path test_results/2867649/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867649/close/fs_test_${SLURM_JOB_ID}.txt
#
#echo "test 2867649 model open set"
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/2867649/model_best.pth.tar --output_path test_results/2867649/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867649/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label2.txt --resume model_save/res101/2867649/model_best.pth.tar --output_path test_results/2867649/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867649/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label3.txt --resume model_save/res101/2867649/model_best.pth.tar --output_path test_results/2867649/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867649/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label4.txt --resume model_save/res101/2867649/model_best.pth.tar --output_path test_results/2867649/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867649/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2867895 model closed set"
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/2867895/model_best.pth.tar --output_path test_results/2867895/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867895/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label2.txt --resume model_save/res101/2867895/model_best.pth.tar --output_path test_results/2867895/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867895/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label3.txt --resume model_save/res101/2867895/model_best.pth.tar --output_path test_results/2867895/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867895/close/nt_test_${SLURM_JOB_ID}.txt
#
#echo "test 2867895 model open set"
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/2867895/model_best.pth.tar --output_path test_results/2867895/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867895/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label2.txt --resume model_save/res101/2867895/model_best.pth.tar --output_path test_results/2867895/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867895/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label4.txt --resume model_save/res101/2867895/model_best.pth.tar --output_path test_results/2867895/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867895/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label3.txt --resume model_save/res101/2867895/model_best.pth.tar --output_path test_results/2867895/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2867895/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2868175 model closed set"
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/2868175/model_best.pth.tar --output_path test_results/2868175/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868175/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label2.txt --resume model_save/res101/2868175/model_best.pth.tar --output_path test_results/2868175/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868175/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label3.txt --resume model_save/res101/2868175/model_best.pth.tar --output_path test_results/2868175/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868175/close/nt_test_${SLURM_JOB_ID}.txt
#
#echo "test 2868175 model open set"
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/2868175/model_best.pth.tar --output_path test_results/2868175/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868175/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label4.txt --resume model_save/res101/2868175/model_best.pth.tar --output_path test_results/2868175/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868175/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label2.txt --resume model_save/res101/2868175/model_best.pth.tar --output_path test_results/2868175/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868175/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label3.txt --resume model_save/res101/2868175/model_best.pth.tar --output_path test_results/2868175/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868175/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2868176 model closed set"
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label1.txt --resume model_save/res101/2868176/model_best.pth.tar --output_path test_results/2868176/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868176/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 4 --test_list data_list/FF++_c40_FS_label2.txt --resume model_save/res101/2868176/model_best.pth.tar --output_path test_results/2868176/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868176/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 4 --test_list data_list/FF++_c40_NT_label3.txt --resume model_save/res101/2868176/model_best.pth.tar --output_path test_results/2868176/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868176/close/nt_test_${SLURM_JOB_ID}.txt
#
#echo "test 2868176 model open set"
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label4.txt --resume model_save/res101/2868176/model_best.pth.tar --output_path test_results/2868176/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868176/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label1.txt --resume model_save/res101/2868176/model_best.pth.tar --output_path test_results/2868176/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868176/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label2.txt --resume model_save/res101/2868176/model_best.pth.tar --output_path test_results/2868176/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868176/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py --image_loader nine_channels -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label3.txt --resume model_save/res101/2868176/model_best.pth.tar --output_path test_results/2868176/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868176/open/nt_test_${SLURM_JOB_ID}.txt




#echo "test 2868242 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/2868242/model_best.pth.tar --output_path test_results/2868242/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868242/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label2.txt --resume model_save/res101/2868242/model_best.pth.tar --output_path test_results/2868242/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868242/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label3.txt --resume model_save/res101/2868242/model_best.pth.tar --output_path test_results/2868242/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868242/close/fs_test_${SLURM_JOB_ID}.txt
#
#echo "test 2868242 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/2868242/model_best.pth.tar --output_path test_results/2868242/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868242/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label2.txt --resume model_save/res101/2868242/model_best.pth.tar --output_path test_results/2868242/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868242/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label3.txt --resume model_save/res101/2868242/model_best.pth.tar --output_path test_results/2868242/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868242/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label4.txt --resume model_save/res101/2868242/model_best.pth.tar --output_path test_results/2868242/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868242/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2868243 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/2868243/model_best.pth.tar --output_path test_results/2868243/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868243/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label2.txt --resume model_save/res101/2868243/model_best.pth.tar --output_path test_results/2868243/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868243/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label3.txt --resume model_save/res101/2868243/model_best.pth.tar --output_path test_results/2868243/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868243/close/nt_test_${SLURM_JOB_ID}.txt
#
#echo "test 2868243 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/2868243/model_best.pth.tar --output_path test_results/2868243/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868243/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label2.txt --resume model_save/res101/2868243/model_best.pth.tar --output_path test_results/2868243/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868243/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label4.txt --resume model_save/res101/2868243/model_best.pth.tar --output_path test_results/2868243/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868243/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label3.txt --resume model_save/res101/2868243/model_best.pth.tar --output_path test_results/2868243/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868243/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2868244 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/2868244/model_best.pth.tar --output_path test_results/2868244/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868244/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label2.txt --resume model_save/res101/2868244/model_best.pth.tar --output_path test_results/2868244/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868244/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label3.txt --resume model_save/res101/2868244/model_best.pth.tar --output_path test_results/2868244/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868244/close/nt_test_${SLURM_JOB_ID}.txt
#
#echo "test 2868244 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/2868244/model_best.pth.tar --output_path test_results/2868244/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868244/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label4.txt --resume model_save/res101/2868244/model_best.pth.tar --output_path test_results/2868244/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868244/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label2.txt --resume model_save/res101/2868244/model_best.pth.tar --output_path test_results/2868244/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868244/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label3.txt --resume model_save/res101/2868244/model_best.pth.tar --output_path test_results/2868244/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868244/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2868245 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label1.txt --resume model_save/res101/2868245/model_best.pth.tar --output_path test_results/2868245/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868245/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label2.txt --resume model_save/res101/2868245/model_best.pth.tar --output_path test_results/2868245/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868245/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label3.txt --resume model_save/res101/2868245/model_best.pth.tar --output_path test_results/2868245/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868245/close/nt_test_${SLURM_JOB_ID}.txt
#
#echo "test 2868245 model open set"
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_DF_label4.txt --resume model_save/res101/2868245/model_best.pth.tar --output_path test_results/2868245/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868245/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_F2F_label1.txt --resume model_save/res101/2868245/model_best.pth.tar --output_path test_results/2868245/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868245/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_FS_label2.txt --resume model_save/res101/2868245/model_best.pth.tar --output_path test_results/2868245/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868245/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c40_NT_label3.txt --resume model_save/res101/2868245/model_best.pth.tar --output_path test_results/2868245/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2868245/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3209738 model closed set"
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/3209738/model_best.pth.tar --output_path test_results/3209738/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209738/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c40_F2F_label2.txt --resume model_save/res101/3209738/model_best.pth.tar --output_path test_results/3209738/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209738/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c40_FS_label3.txt --resume model_save/res101/3209738/model_best.pth.tar --output_path test_results/3209738/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209738/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c40_NT_label4.txt --resume model_save/res101/3209738/model_best.pth.tar --output_path test_results/3209738/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209738/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3209771 model closed set"
#python -u test.py -b 1 --n_classes_total 5 --test_list data_list/FF++_c40_DF_label1.txt --resume model_save/res101/3209771/model_best.pth.tar --output_path test_results/3209771/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209771/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 5 --test_list data_list/FF++_c40_F2F_label2.txt --resume model_save/res101/3209771/model_best.pth.tar --output_path test_results/3209771/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209771/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 5 --test_list data_list/FF++_c40_FS_label3.txt --resume model_save/res101/3209771/model_best.pth.tar --output_path test_results/3209771/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209771/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 5 --test_list data_list/FF++_c40_NT_label4.txt --resume model_save/res101/3209771/model_best.pth.tar --output_path test_results/3209771/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209771/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3209679 model closed set"
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3209679/model_best.pth.tar --output_path test_results/3209679/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209679/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3209679/model_best.pth.tar --output_path test_results/3209679/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209679/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3209679/model_best.pth.tar --output_path test_results/3209679/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209679/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3209679/model_best.pth.tar --output_path test_results/3209679/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209679/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3209785 model closed set"
#python -u test.py -b 1 --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3209785/model_best.pth.tar --output_path test_results/3209785/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209785/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3209785/model_best.pth.tar --output_path test_results/3209785/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209785/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3209785/model_best.pth.tar --output_path test_results/3209785/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209785/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3209785/model_best.pth.tar --output_path test_results/3209785/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3209785/close/nt_test_${SLURM_JOB_ID}.txt
#echo "here"
#python -u test.py -b 1 --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3209785/model_best.pth.tar --output_path test_results/3209785/close/df_test_detail.txt| tee test_results/3209785/close/df_test.txt
#python -u test.py -b 1 --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3209785/model_best.pth.tar --output_path test_results/3209785/close/f2f_test_detail.txt| tee test_results/3209785/close/f2f_test.txt
#python -u test.py -b 1 --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3209785/model_best.pth.tar --output_path test_results/3209785/close/fs_test_detail.txt| tee test_results/3209785/close/fs_test.txt
#python -u test.py -b 1 --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3209785/model_best.pth.tar --output_path test_results/3209785/close/nt_test_detail.txt| tee test_results/3209785/close/nt_test.txt


#echo "test 3327296 model closed set"
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3327296/model_best.pth.tar --output_path test_results/3327296/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327296/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3327296/model_best.pth.tar --output_path test_results/3327296/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327296/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3327296/model_best.pth.tar --output_path test_results/3327296/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327296/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3327296/model_best.pth.tar --output_path test_results/3327296/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327296/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3327301 model closed set"
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3327301/model_best.pth.tar --output_path test_results/3327301/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327301/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3327301/model_best.pth.tar --output_path test_results/3327301/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327301/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3327301/model_best.pth.tar --output_path test_results/3327301/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327301/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3327301/model_best.pth.tar --output_path test_results/3327301/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327301/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3327311 model closed set"
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3327311/model_best.pth.tar --output_path test_results/3327311/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327311/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3327311/model_best.pth.tar --output_path test_results/3327311/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327311/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3327311/model_best.pth.tar --output_path test_results/3327311/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327311/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3327311/model_best.pth.tar --output_path test_results/3327311/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327311/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3327312 model closed set"
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3327312/model_best.pth.tar --output_path test_results/3327312/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327312/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3327312/model_best.pth.tar --output_path test_results/3327312/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327312/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3327312/model_best.pth.tar --output_path test_results/3327312/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327312/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3327312/model_best.pth.tar --output_path test_results/3327312/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327312/close/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3327325 model closed set"
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3327325/model_best.pth.tar --output_path test_results/3327325/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327325/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3327325/model_best.pth.tar --output_path test_results/3327325/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327325/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3327325/model_best.pth.tar --output_path test_results/3327325/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327325/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3327325/model_best.pth.tar --output_path test_results/3327325/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327325/close/nt_test_${SLURM_JOB_ID}.txt

#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3327325/model_best.pth.tar --output_path test_results/3327325/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327325/open/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3327325/model_best.pth.tar --output_path test_results/3327325/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327325/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3327325/model_best.pth.tar --output_path test_results/3327325/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327325/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3327325/model_best.pth.tar --output_path test_results/3327325/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327325/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3327330 model closed set"
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3327330/model_best.pth.tar --output_path test_results/3327330/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327330/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3327330/model_best.pth.tar --output_path test_results/3327330/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327330/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3327330/model_best.pth.tar --output_path test_results/3327330/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327330/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3327330/model_best.pth.tar --output_path test_results/3327330/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327330/close/nt_test_${SLURM_JOB_ID}.txt

#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3327330/model_best.pth.tar --output_path test_results/3327330/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327330/open/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3327330/model_best.pth.tar --output_path test_results/3327330/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327330/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3327330/model_best.pth.tar --output_path test_results/3327330/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327330/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3327330/model_best.pth.tar --output_path test_results/3327330/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3327330/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3361491 model closed set"
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5  --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3361491/model_best.pth.tar --output_path test_results/3361491/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3361491/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5  --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3361491/model_best.pth.tar --output_path test_results/3361491/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3361491/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5  --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3361491/model_best.pth.tar --output_path test_results/3361491/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3361491/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5  --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3361491/model_best.pth.tar --output_path test_results/3361491/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3361491/close/nt_test_${SLURM_JOB_ID}.txt
#echo "test 3361508 model closed set"
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5  --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3361508/model_best.pth.tar --output_path test_results/3361508/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3361508/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5  --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3361508/model_best.pth.tar --output_path test_results/3361508/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3361508/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5  --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3361508/model_best.pth.tar --output_path test_results/3361508/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3361508/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5  --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3361508/model_best.pth.tar --output_path test_results/3361508/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3361508/close/nt_test_${SLURM_JOB_ID}.txt
#echo "test 3410621 model closed set"
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5  --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3410621/model_best.pth.tar --output_path test_results/3410621/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3410621/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5  --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3410621/model_best.pth.tar --output_path test_results/3410621/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3410621/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5  --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3410621/model_best.pth.tar --output_path test_results/3410621/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3410621/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5  --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3410621/model_best.pth.tar --output_path test_results/3410621/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3410621/close/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3367609 model closed set"
#python -u test.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3367609/model_best.pth.tar --output_path test_results/3367609/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3367609/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3367609/model_best.pth.tar --output_path test_results/3367609/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3367609/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3367609/model_best.pth.tar --output_path test_results/3367609/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3367609/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3367609/model_best.pth.tar --output_path test_results/3367609/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3367609/close/nt_test_${SLURM_JOB_ID}.txt



#echo "test 3389506 model closed set"
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3389506/model_best.pth.tar --output_path test_results/3389506/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389506/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3389506/model_best.pth.tar --output_path test_results/3389506/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389506/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3389506/model_best.pth.tar --output_path test_results/3389506/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389506/close/nt_test_${SLURM_JOB_ID}.txt
#
#echo "test 3389506 model open set"
#python -u test_open_set.py -b 1 --image_loader lab_loader --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3389506/model_best.pth.tar --output_path test_results/3389506/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389506/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --image_loader lab_loader --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/3389506/model_best.pth.tar --output_path test_results/3389506/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389506/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --image_loader lab_loader --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/3389506/model_best.pth.tar --output_path test_results/3389506/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389506/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --image_loader lab_loader --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3389506/model_best.pth.tar --output_path test_results/3389506/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389506/open/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3389498 model closed set"
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3389498/model_best.pth.tar --output_path test_results/3389498/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389498/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3389498/model_best.pth.tar --output_path test_results/3389498/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389498/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3389498/model_best.pth.tar --output_path test_results/3389498/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389498/close/nt_test_${SLURM_JOB_ID}.txt
#
#echo "test 3389498 model open set"
#python -u test_open_set.py -b 1 --image_loader hsv_loader --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3389498/model_best.pth.tar --output_path test_results/3389498/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389498/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --image_loader hsv_loader --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/3389498/model_best.pth.tar --output_path test_results/3389498/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389498/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --image_loader hsv_loader --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/3389498/model_best.pth.tar --output_path test_results/3389498/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389498/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --image_loader hsv_loader --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3389498/model_best.pth.tar --output_path test_results/3389498/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3389498/open/nt_test_${SLURM_JOB_ID}.txt




#echo "test 3444074 model closed set"
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444074/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444074/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444074/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444074/close/nt_test_${SLURM_JOB_ID}.txt
#echo "test 3444074 model open set"
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/open/df_test_detail.txt| tee test_results/3444074/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/open/f2f_test_detail.txt| tee test_results/3444074/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/open/fs_test_detail.txt| tee test_results/3444074/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/open/nt_test_detail.txt| tee test_results/3444074/open/nt_test_${SLURM_JOB_ID}.txt

#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/3444074/model_best.pth.tar --output_path test_results/3444074/DFDC/DFDC_test_detail.txt




#echo "test 3409590 model closed set"
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3409590/model_best.pth.tar --output_path test_results/3409590/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3409590/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3409590/model_best.pth.tar --output_path test_results/3409590/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3409590/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3409590/model_best.pth.tar --output_path test_results/3409590/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3409590/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3409590/model_best.pth.tar --output_path test_results/3409590/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3409590/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3409598 model closed set"
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3409598/model_best.pth.tar --output_path test_results/3409598/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3409598/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3409598/model_best.pth.tar --output_path test_results/3409598/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3409598/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3409598/model_best.pth.tar --output_path test_results/3409598/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3409598/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --image_loader lab_loader --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3409598/model_best.pth.tar --output_path test_results/3409598/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3409598/close/nt_test_${SLURM_JOB_ID}.txt



#echo "test 3442823 model closed set"
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels  --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3442823/model_best.pth.tar --output_path test_results/3442823/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442823/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels  --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3442823/model_best.pth.tar --output_path test_results/3442823/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442823/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels  --n_classes_total 4 --test_list data_list/FF++_c23_NT_label3.txt --resume model_save/res101/3442823/model_best.pth.tar --output_path test_results/3442823/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442823/close/nt_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels  --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3442823/model_best.pth.tar --output_path test_results/3442823/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442823/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels  --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3442823/model_best.pth.tar --output_path test_results/3442823/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442823/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels  --n_classes_total 4 --test_list data_list/FF++_c23_FS_label4.txt --resume model_save/res101/3442823/model_best.pth.tar --output_path test_results/3442823/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442823/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels  --n_classes_total 4 --test_list data_list/FF++_c23_NT_label3.txt --resume model_save/res101/3442823/model_best.pth.tar --output_path test_results/3442823/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442823/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3456390 model closed set"
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3456390/model_best.pth.tar --output_path test_results/3456390/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456390/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3456390/model_best.pth.tar --output_path test_results/3456390/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456390/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c23_NT_label3.txt --resume model_save/res101/3456390/model_best.pth.tar --output_path test_results/3456390/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456390/close/nt_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3456390/model_best.pth.tar --output_path test_results/3456390/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456390/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3456390/model_best.pth.tar --output_path test_results/3456390/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456390/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c23_FS_label4.txt --resume model_save/res101/3456390/model_best.pth.tar --output_path test_results/3456390/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456390/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --n_classes_total 4 --test_list data_list/FF++_c23_NT_label3.txt --resume model_save/res101/3456390/model_best.pth.tar --output_path test_results/3456390/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456390/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3456382 model closed set"
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3456382/model_best.pth.tar --output_path test_results/3456382/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456382/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3456382/model_best.pth.tar --output_path test_results/3456382/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456382/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/FF++_c23_NT_label3.txt --resume model_save/res101/3456382/model_best.pth.tar --output_path test_results/3456382/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456382/close/nt_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3456382/model_best.pth.tar --output_path test_results/3456382/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456382/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3456382/model_best.pth.tar --output_path test_results/3456382/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456382/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/FF++_c23_FS_label4.txt --resume model_save/res101/3456382/model_best.pth.tar --output_path test_results/3456382/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456382/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/FF++_c23_NT_label3.txt --resume model_save/res101/3456382/model_best.pth.tar --output_path test_results/3456382/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3456382/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3445186 model closed set"
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3445186/model_best.pth.tar --output_path test_results/3445186/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3445186/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3445186/model_best.pth.tar --output_path test_results/3445186/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3445186/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3445186/model_best.pth.tar --output_path test_results/3445186/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3445186/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3445186/model_best.pth.tar --output_path test_results/3445186/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3445186/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3445192 model closed set"
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3445192/model_best.pth.tar --output_path test_results/3445192/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3445192/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3445192/model_best.pth.tar --output_path test_results/3445192/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3445192/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3445192/model_best.pth.tar --output_path test_results/3445192/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3445192/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3445192/model_best.pth.tar --output_path test_results/3445192/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3445192/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3445176 model closed set"
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3445176/model_best.pth.tar --output_path test_results/3445176/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3445176/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3445176/model_best.pth.tar --output_path test_results/3445176/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3445176/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3445176/model_best.pth.tar --output_path test_results/3445176/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3445176/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_6ch --image_loader six_channels --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3445176/model_best.pth.tar --output_path test_results/3445176/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3445176/close/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3453160 model closed set"
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3453160/model_best.pth.tar --output_path test_results/3453160/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3453160/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3453160/model_best.pth.tar --output_path test_results/3453160/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3453160/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3453160/model_best.pth.tar --output_path test_results/3453160/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3453160/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3453160/model_best.pth.tar --output_path test_results/3453160/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3453160/close/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3455067 model closed set"
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3455067/model_best.pth.tar --output_path test_results/3455067/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455067/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3455067/model_best.pth.tar --output_path test_results/3455067/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455067/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3455067/model_best.pth.tar --output_path test_results/3455067/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455067/close/fs_test_${SLURM_JOB_ID}.txt
#                                                           
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3455067/model_best.pth.tar --output_path test_results/3455067/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455067/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3455067/model_best.pth.tar --output_path test_results/3455067/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455067/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3455067/model_best.pth.tar --output_path test_results/3455067/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455067/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3455067/model_best.pth.tar --output_path test_results/3455067/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455067/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3455070 model closed set"
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3455070/model_best.pth.tar --output_path test_results/3455070/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455070/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3455070/model_best.pth.tar --output_path test_results/3455070/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455070/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3455070/model_best.pth.tar --output_path test_results/3455070/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455070/close/nt_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3455070/model_best.pth.tar --output_path test_results/3455070/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455070/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3455070/model_best.pth.tar --output_path test_results/3455070/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455070/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label4.txt --resume model_save/res101/3455070/model_best.pth.tar --output_path test_results/3455070/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455070/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3455070/model_best.pth.tar --output_path test_results/3455070/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455070/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3455072 model closed set"
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3455072/model_best.pth.tar --output_path test_results/3455072/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455072/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/3455072/model_best.pth.tar --output_path test_results/3455072/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455072/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3455072/model_best.pth.tar --output_path test_results/3455072/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455072/close/nt_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3455072/model_best.pth.tar --output_path test_results/3455072/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455072/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/3455072/model_best.pth.tar --output_path test_results/3455072/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455072/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/3455072/model_best.pth.tar --output_path test_results/3455072/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455072/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3455072/model_best.pth.tar --output_path test_results/3455072/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455072/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3455073 model closed set"
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label1.txt --resume model_save/res101/3455073/model_best.pth.tar --output_path test_results/3455073/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455073/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/3455073/model_best.pth.tar --output_path test_results/3455073/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455073/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3455073/model_best.pth.tar --output_path test_results/3455073/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455073/close/nt_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label4.txt --resume model_save/res101/3455073/model_best.pth.tar --output_path test_results/3455073/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455073/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label1.txt --resume model_save/res101/3455073/model_best.pth.tar --output_path test_results/3455073/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455073/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/3455073/model_best.pth.tar --output_path test_results/3455073/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455073/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name res101_9ch --image_loader nine_channels --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3455073/model_best.pth.tar --output_path test_results/3455073/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3455073/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3444054 model closed set"
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3444054/model_best.pth.tar --output_path test_results/3444054/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444054/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3444054/model_best.pth.tar --output_path test_results/3444054/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444054/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3444054/model_best.pth.tar --output_path test_results/3444054/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444054/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3444054/model_best.pth.tar --output_path test_results/3444054/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444054/close/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3442845 model closed set"
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3442845/model_best.pth.tar --output_path test_results/3442845/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442845/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3442845/model_best.pth.tar --output_path test_results/3442845/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442845/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3442845/model_best.pth.tar --output_path test_results/3442845/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442845/close/nt_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3442845/model_best.pth.tar --output_path test_results/3442845/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442845/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3442845/model_best.pth.tar --output_path test_results/3442845/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442845/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label4.txt --resume model_save/res101/3442845/model_best.pth.tar --output_path test_results/3442845/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442845/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3442845/model_best.pth.tar --output_path test_results/3442845/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3442845/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3443982 model closed set"
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3443982/model_best.pth.tar --output_path test_results/3443982/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3443982/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3443982/model_best.pth.tar --output_path test_results/3443982/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3443982/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3443982/model_best.pth.tar --output_path test_results/3443982/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3443982/close/nt_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3443982/model_best.pth.tar --output_path test_results/3443982/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3443982/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3443982/model_best.pth.tar --output_path test_results/3443982/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3443982/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/faceswap_test_800999_frame30_label4.txt --resume model_save/res101/3443982/model_best.pth.tar --output_path test_results/3443982/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3443982/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/3443982/model_best.pth.tar --output_path test_results/3443982/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3443982/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 3499074 model closed set"
#python -u test.py -b 1  --image_loader single_channel --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3499074/model_best.pth.tar --output_path test_results/3499074/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3499074/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1  --image_loader single_channel --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3499074/model_best.pth.tar --output_path test_results/3499074/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3499074/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1  --image_loader single_channel --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3499074/model_best.pth.tar --output_path test_results/3499074/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3499074/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1  --image_loader single_channel --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3499074/model_best.pth.tar --output_path test_results/3499074/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3499074/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3499072 model closed set"
#python -u test.py -b 1  --image_loader single_channel --n_classes_total 5 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/3499072/model_best.pth.tar --output_path test_results/3499072/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3499072/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1  --image_loader single_channel --n_classes_total 5 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/3499072/model_best.pth.tar --output_path test_results/3499072/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3499072/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1  --image_loader single_channel --n_classes_total 5 --test_list data_list/faceswap_test_800999_frame30_label3.txt --resume model_save/res101/3499072/model_best.pth.tar --output_path test_results/3499072/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3499072/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1  --image_loader single_channel --n_classes_total 5 --test_list data_list/neuraltextures_test_800999_frame30_label4.txt --resume model_save/res101/3499072/model_best.pth.tar --output_path test_results/3499072/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3499072/close/nt_test_${SLURM_JOB_ID}.txt



#echo "cross-datasets 3543488 test"
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/DFDC/DFDC_test_detail.txt


#echo "cross-datasets 3543490 test"
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/3543490/model_best.pth.tar --output_path test_results/3543490/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/3543490/model_best.pth.tar --output_path test_results/3543490/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/3543490/model_best.pth.tar --output_path test_results/3543490/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/3543490/model_best.pth.tar --output_path test_results/3543490/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/3543490/model_best.pth.tar --output_path test_results/3543490/DFDC/DFDC_test_detail.txt
#echo "cross-datasets 3556841 test"
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/DFDC/DFDC_test_detail.txt

#echo "cross-datasets 3543492 test"
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/DFDC/DFDC_test_detail.txt



#echo "test 3543488 model closed set"
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5  --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/close/df_test_detail.txt| tee test_results/3543488/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5  --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/close/f2f_test_detail.txt| tee test_results/3543488/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5  --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/close/fs_test_detail.txt| tee test_results/3543488/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5  --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/close/nt_test_detail.txt| tee test_results/3543488/close/nt_test_${SLURM_JOB_ID}.txt
#echo "test 3543488 model open set"
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5  --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/open/df_test_detail.txt| tee test_results/3543488/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5  --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/open/f2f_test_detail.txt| tee test_results/3543488/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5  --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/open/fs_test_detail.txt| tee test_results/3543488/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5  --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/open/nt_test_detail.txt| tee test_results/3543488/open/nt_test_${SLURM_JOB_ID}.txt

#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/3543488/model_best.pth.tar --output_path test_results/3543488/DFDC/DFDC_test_detail.txt




#echo "test 3543490 model closed set"
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5  --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3543490/model_best.pth.tar --output_path test_results/3543490/close/df_test_detail.txt| tee test_results/3543490/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5  --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3543490/model_best.pth.tar --output_path test_results/3543490/close/f2f_test_detail.txt| tee test_results/3543490/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5  --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3543490/model_best.pth.tar --output_path test_results/3543490/close/fs_test_detail.txt| tee test_results/3543490/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_lab --n_classes_total 5  --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3543490/model_best.pth.tar --output_path test_results/3543490/close/nt_test_detail.txt| tee test_results/3543490/close/nt_test_${SLURM_JOB_ID}.txt



#echo "test 3556841 model closed set"
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5  --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/close/df_test_detail.txt| tee test_results/3556841/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5  --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/close/f2f_test_detail.txt| tee test_results/3556841/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5  --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/close/fs_test_detail.txt| tee test_results/3556841/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5  --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/close/nt_test_detail.txt| tee test_results/3556841/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3556841 model open set"
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5  --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/open/df_test_detail.txt| tee test_results/3556841/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5  --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/open/f2f_test_detail.txt| tee test_results/3556841/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5  --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/open/fs_test_detail.txt| tee test_results/3556841/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5  --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/open/nt_test_detail.txt| tee test_results/3556841/open/nt_test_${SLURM_JOB_ID}.txt

#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_hsv --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/3556841/model_best.pth.tar --output_path test_results/3556841/DFDC/DFDC_test_detail.txt



#echo "test 3543492 model closed set"
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5  --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/close/df_test_detail.txt| tee test_results/3543492/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5  --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/close/f2f_test_detail.txt| tee test_results/3543492/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5  --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/close/fs_test_detail.txt| tee test_results/3543492/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5  --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/close/nt_test_detail.txt| tee test_results/3543492/close/nt_test_${SLURM_JOB_ID}.txt
#echo "test 3543492 model open set"
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5  --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/open/df_test_detail.txt| tee test_results/3543492/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5  --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/open/f2f_test_detail.txt| tee test_results/3543492/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5  --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/open/fs_test_detail.txt| tee test_results/3543492/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5  --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/open/nt_test_detail.txt| tee test_results/3543492/open/nt_test_${SLURM_JOB_ID}.txt

#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_6channels --image_loader rgb_ycbcr --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/3543492/model_best.pth.tar --output_path test_results/3543492/DFDC/DFDC_test_detail.txt




#echo "test 3575187 model closed set"
#python -u test.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/close/df_test_detail.txt| tee test_results/3575187/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/close/f2f_test_detail.txt| tee test_results/3575187/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/close/nt_test_detail.txt| tee test_results/3575187/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/close/nt_test_detail.txt| tee test_results/3575187/close/nt_test_${SLURM_JOB_ID}.txt
#
#echo "test 3575187 model open set"
#python -u test_open_set.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/open/df_test_detail.txt| tee test_results/3575187/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/open/f2f_test_detail.txt| tee test_results/3575187/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/open/fs_test_detail.txt| tee test_results/3575187/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/open/nt_test_detail.txt| tee test_results/3575187/open/nt_test_${SLURM_JOB_ID}.txt

#python -u test_open_set.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader hsv_loader --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/3575187/model_best.pth.tar --output_path test_results/3575187/DFDC/DFDC_test_detail.txt




#echo "test 3575183 model closed set"
#python -u test.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/close/df_test_detail.txt| tee test_results/3575183/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/close/f2f_test_detail.txt| tee test_results/3575183/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/close/nt_test_detail.txt| tee test_results/3575183/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/close/nt_test_detail.txt| tee test_results/3575183/close/nt_test_${SLURM_JOB_ID}.txt
#
#echo "test 3575183 model open set"
#python -u test_open_set.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/open/df_test_detail.txt| tee test_results/3575183/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/open/f2f_test_detail.txt| tee test_results/3575183/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/open/fs_test_detail.txt| tee test_results/3575183/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/open/nt_test_detail.txt| tee test_results/3575183/open/nt_test_${SLURM_JOB_ID}.txt

#python -u test_open_set.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader lab_loader --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/3575183/model_best.pth.tar --output_path test_results/3575183/DFDC/DFDC_test_detail.txt



#echo "test 3575287 model closed set"
#python -u test.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/close/df_test_detail.txt| tee test_results/3575287/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/close/f2f_test_detail.txt| tee test_results/3575287/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/close/nt_test_detail.txt| tee test_results/3575287/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/close/nt_test_detail.txt| tee test_results/3575287/close/nt_test_${SLURM_JOB_ID}.txt
#
#echo "test 3575287 model open set"
#python -u test_open_set.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/open/df_test_detail.txt| tee test_results/3575287/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/open/f2f_test_detail.txt| tee test_results/3575287/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/open/fs_test_detail.txt| tee test_results/3575287/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/open/nt_test_detail.txt| tee test_results/3575287/open/nt_test_${SLURM_JOB_ID}.txt

#python -u test_open_set.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --image_loader ycbcr_loader --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/3575287/model_best.pth.tar --output_path test_results/3575287/DFDC/DFDC_test_detail.txt





#echo "test 3444101 model closed set"
#python -u test.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444101/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444101/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/close/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444101/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444101/close/nt_test_${SLURM_JOB_ID}.txt
#
#echo "test 3444101 model open set"
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/open/df_test_detail.txt| tee test_results/3444101/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/open/f2f_test_detail.txt| tee test_results/3444101/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/open/fs_test_detail.txt| tee test_results/3444101/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/open/nt_test_detail.txt| tee test_results/3444101/open/nt_test_${SLURM_JOB_ID}.txt

#echo "test 3444101 cross datasets"
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/3444101/model_best.pth.tar --output_path test_results/3444101/DFDC/DFDC_test_detail.txt

echo "t-sne"
#python -u t-sne.py --model_name xception --n_classes_total 5 --test_list data_list/tsne_FF++_c23_all01234_test_test.txt --resume model_save/res101/3444101/model_best.pth.tar
#python -u tsne_new.py --model_name xception --n_classes_total 5 --test_list data_list/tsne_FF++_c23_all01234_test.txt --resume model_save/res101/3444101/model_best.pth.tar
#python -u tsne_new.py --model_name xception --n_classes_total 5 --test_list data_list/FF++_c23_all01234_test.txt --resume model_save/res101/3444101/model_best.pth.tar
python -u tsne_new.py --n_classes_total 5 --test_list data_list/FF++_c23_all01234_test.txt --resume model_save/res101/3209785/model_best.pth.tar


#echo "test 5243687 model open set"
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/5243687/model_best.pth.tar --output_path test_results/5243687/open/df_test_detail.txt| tee test_results/5243687/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/5243687/model_best.pth.tar --output_path test_results/5243687/open/f2f_test_detail.txt| tee test_results/5243687/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/5243687/model_best.pth.tar --output_path test_results/5243687/open/fs_test_detail.txt| tee test_results/5243687/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/5243687/model_best.pth.tar --output_path test_results/5243687/open/nt_test_detail.txt| tee test_results/5243687/open/nt_test_${SLURM_JOB_ID}.txt

#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/FakeAV_random500.txt --resume model_save/res101/5243687/model_best.pth.tar --output_path test_results/5243687/FakeAV/FakeAV_random500_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/5243687/model_best.pth.tar --output_path test_results/5243687/DFD/DFD_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/KoDF_test_999.txt --resume model_save/res101/5243687/model_best.pth.tar --output_path test_results/5243687/KoDF/KoDF_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/5243687/model_best.pth.tar --output_path test_results/5243687/Celeb/Celeb_test_detail.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --image_loader nine_channels --n_classes_total 5 --test_list data_list/dfdc_test_random2w.txt --resume model_save/res101/5243687/model_best.pth.tar --output_path test_results/5243687/DFDC/DFDC_test_detail.txt








#echo "test 3444115 model closed set"
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3444115/model_best.pth.tar --output_path test_results/3444115/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444115/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3444115/model_best.pth.tar --output_path test_results/3444115/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444115/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_NT_label3.txt --resume model_save/res101/3444115/model_best.pth.tar --output_path test_results/3444115/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444115/close/nt_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3444115/model_best.pth.tar --output_path test_results/3444115/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444115/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3444115/model_best.pth.tar --output_path test_results/3444115/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444115/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_FS_label4.txt --resume model_save/res101/3444115/model_best.pth.tar --output_path test_results/3444115/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444115/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_NT_label3.txt --resume model_save/res101/3444115/model_best.pth.tar --output_path test_results/3444115/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/3444115/open/nt_test_${SLURM_JOB_ID}.txt



#echo "test 3575379 model closed set"
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3575379/model_best.pth.tar --output_path test_results/3575379/close/df_test_detail.txt| tee test_results/3575379/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3575379/model_best.pth.tar --output_path test_results/3575379/close/f2f_test_detail.txt| tee test_results/3575379/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3575379/model_best.pth.tar --output_path test_results/3575379/close/fs_test_detail.txt| tee test_results/3575379/close/fs_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3575379/model_best.pth.tar --output_path test_results/3575379/open/df_test_detail.txt| tee test_results/3575379/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3575379/model_best.pth.tar --output_path test_results/3575379/open/f2f_test_detail.txt| tee test_results/3575379/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3575379/model_best.pth.tar --output_path test_results/3575379/open/fs_test_detail.txt| tee test_results/3575379/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3575379/model_best.pth.tar --output_path test_results/3575379/open/nt_test_detail.txt| tee test_results/3575379/open/nt_test_${SLURM_JOB_ID}.txt



#echo "test 3575378 model closed set"
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3575378/model_best.pth.tar --output_path test_results/3575378/close/df_test_detail.txt| tee test_results/3575378/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_FS_label2.txt --resume model_save/res101/3575378/model_best.pth.tar --output_path test_results/3575378/close/fs_test_detail.txt| tee test_results/3575378/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_NT_label3.txt --resume model_save/res101/3575378/model_best.pth.tar --output_path test_results/3575378/close/nt_test_detail.txt| tee test_results/3575378/close/nt_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3575378/model_best.pth.tar --output_path test_results/3575378/open/df_test_detail.txt| tee test_results/3575378/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label4.txt --resume model_save/res101/3575378/model_best.pth.tar --output_path test_results/3575378/open/f2f_test_detail.txt| tee test_results/3575378/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_FS_label2.txt --resume model_save/res101/3575378/model_best.pth.tar --output_path test_results/3575378/open/fs_test_detail.txt| tee test_results/3575378/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_NT_label3.txt --resume model_save/res101/3575378/model_best.pth.tar --output_path test_results/3575378/open/nt_test_detail.txt| tee test_results/3575378/open/nt_test_${SLURM_JOB_ID}.txt



#echo "test 3575380 model closed set"
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label1.txt --resume model_save/res101/3575380/model_best.pth.tar --output_path test_results/3575380/close/f2f_test_detail.txt| tee test_results/3575380/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_FS_label2.txt --resume model_save/res101/3575380/model_best.pth.tar --output_path test_results/3575380/close/fs_test_detail.txt| tee test_results/3575380/close/fs_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_NT_label3.txt --resume model_save/res101/3575380/model_best.pth.tar --output_path test_results/3575380/close/nt_test_detail.txt| tee test_results/3575380/close/nt_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_DF_label4.txt --resume model_save/res101/3575380/model_best.pth.tar --output_path test_results/3575380/open/df_test_detail.txt| tee test_results/3575380/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label1.txt --resume model_save/res101/3575380/model_best.pth.tar --output_path test_results/3575380/open/f2f_test_detail.txt| tee test_results/3575380/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_FS_label2.txt --resume model_save/res101/3575380/model_best.pth.tar --output_path test_results/3575380/open/fs_test_detail.txt| tee test_results/3575380/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception --n_classes_total 4 --test_list data_list/FF++_c23_NT_label3.txt --resume model_save/res101/3575380/model_best.pth.tar --output_path test_results/3575380/open/nt_test_detail.txt| tee test_results/3575380/open/nt_test_${SLURM_JOB_ID}.txt




#echo "test 3575375 model closed set"
#python -u test.py -b 1 --model_name xception_9channels --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3575375/model_best.pth.tar --output_path test_results/3575375/close/df_test_detail.txt| tee test_results/3575375/close/df_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_9channels --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3575375/model_best.pth.tar --output_path test_results/3575375/close/f2f_test_detail.txt| tee test_results/3575375/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test.py -b 1 --model_name xception_9channels --n_classes_total 4 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3575375/model_best.pth.tar --output_path test_results/3575375/close/fs_test_detail.txt| tee test_results/3575375/close/fs_test_${SLURM_JOB_ID}.txt
#
#python -u test_open_set.py -b 1 --model_name xception_9channels --n_classes_total 4 --test_list data_list/FF++_c23_DF_label1.txt --resume model_save/res101/3575375/model_best.pth.tar --output_path test_results/3575375/open/df_test_detail.txt| tee test_results/3575375/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --n_classes_total 4 --test_list data_list/FF++_c23_F2F_label2.txt --resume model_save/res101/3575375/model_best.pth.tar --output_path test_results/3575375/open/f2f_test_detail.txt| tee test_results/3575375/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --n_classes_total 4 --test_list data_list/FF++_c23_FS_label3.txt --resume model_save/res101/3575375/model_best.pth.tar --output_path test_results/3575375/open/fs_test_detail.txt| tee test_results/3575375/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_open_set.py -b 1 --model_name xception_9channels --n_classes_total 4 --test_list data_list/FF++_c23_NT_label4.txt --resume model_save/res101/3575375/model_best.pth.tar --output_path test_results/3575375/open/nt_test_detail.txt| tee test_results/3575375/open/nt_test_${SLURM_JOB_ID}.txt







