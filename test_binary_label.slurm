#!/bin/bash
#SBATCH --job-name="API-test"   # Sensible name for the job
#SBATCH --account=ie-idi   # Account for consumed resources
#SBATCH --nodes=1           # Allocate 1 nodes for the job
#SBATCH -c 8                 # Number of cores (can vary)
#SBATCH --gres=gpu:V100
#SBATCH --time=99-00:00:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --partition=GPUQ
#SBATCH --mail-user=ying.xu@ntnu.no
#SBATCH --ntasks-per-node=1     # 1 task per node
#SBATCH --nice=2147483645
#SBATCH --priority=TOP
#SBATCH --no-requeue
module purge
cd ${SLURM_SUBMIT_DIR}
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

#module load intel/2020b
module load Python/3.8.6-GCCcore-10.2.0
source venv/bin/activate


#echo "test 2569121 model closed set"
#python -u test_binary.py -b 1 --n_classes_total 2 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2569121/model_best.pth.tar --output_path test_results/2569121/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569121/close/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 2 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2569121/model_best.pth.tar --output_path test_results/2569121/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569121/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 2 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2569121/model_best.pth.tar --output_path test_results/2569121/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569121/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2569121 model open set"
#python -u test_binary.py -b 1 --n_classes_total 2 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2569121/model_best.pth.tar --output_path test_results/2569121/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569121/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 2 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/2569121/model_best.pth.tar --output_path test_results/2569121/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569121/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 2 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2569121/model_best.pth.tar --output_path test_results/2569121/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569121/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py -b 1 --n_classes_total 2 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2569121/model_best.pth.tar --output_path test_results/2569121/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569121/open/nt_test_${SLURM_JOB_ID}.txt


#echo "test 2569123 model closed set"
#python -u test_binary.py --image_loader bvcb_loader -b 1 --n_classes_total 2 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2569123/model_best.pth.tar --output_path test_results/2569123/close/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569123/close/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py --image_loader bvcb_loader -b 1 --n_classes_total 2 --test_list data_list/face2face_test_800999_frame30_label2.txt --resume model_save/res101/2569123/model_best.pth.tar --output_path test_results/2569123/close/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569123/close/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py --image_loader bvcb_loader -b 1 --n_classes_total 2 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2569123/model_best.pth.tar --output_path test_results/2569123/close/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569123/close/nt_test_${SLURM_JOB_ID}.txt

#echo "test 2569123 model open set"
#python -u test_binary.py --image_loader bvcb_loader -b 1 --n_classes_total 2 --test_list data_list/deepfakes_test_800999_frame30_label1.txt --resume model_save/res101/2569123/model_best.pth.tar --output_path test_results/2569123/open/df_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569123/open/df_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py --image_loader bvcb_loader -b 1 --n_classes_total 2 --test_list data_list/face2face_test_800999_frame30_label4.txt --resume model_save/res101/2569123/model_best.pth.tar --output_path test_results/2569123/open/f2f_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569123/open/f2f_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py --image_loader bvcb_loader -b 1 --n_classes_total 2 --test_list data_list/faceswap_test_800999_frame30_label2.txt --resume model_save/res101/2569123/model_best.pth.tar --output_path test_results/2569123/open/fs_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569123/open/fs_test_${SLURM_JOB_ID}.txt
#python -u test_binary.py --image_loader bvcb_loader -b 1 --n_classes_total 2 --test_list data_list/neuraltextures_test_800999_frame30_label3.txt --resume model_save/res101/2569123/model_best.pth.tar --output_path test_results/2569123/open/nt_test_detail_${SLURM_JOB_ID}_model.txt| tee test_results/2569123/open/nt_test_${SLURM_JOB_ID}.txt
