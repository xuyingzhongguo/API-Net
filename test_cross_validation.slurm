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

#echo "test 2475004 model on Celeb-DF-v2"
#python -u test_open_set.py -b 1 --n_classes_total 5 --test_list data_list/celeb_df_test.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path cross_val_test_results/celeb-df/test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee cross_val_test_results/celeb-df/test_${SLURM_JOB_ID}.txt

#echo "test 2475004 model on DFDC_val"
#python -u test_open_set.py -b 1 --n_classes_total 5 --test_list data_list/dfdc_val.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path cross_val_test_results/dfdc/test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee cross_val_test_results/dfdc/test_${SLURM_JOB_ID}.txt

#echo "test 2475004 model on Deeperf_test"
#python -u test_open_set.py -b 1 --n_classes_total 5 --test_list data_list/deeperf_test.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path cross_val_test_results/deeperf/test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee cross_val_test_results/deeperf/test_${SLURM_JOB_ID}.txt

#echo "test 2475004 model on DFD_test"
#python -u test_open_set.py -b 1 --n_classes_total 5 --test_list data_list/dfd_test.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path cross_val_test_results/deepfakedetection/test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee cross_val_test_results/deepfakedetection/test_${SLURM_JOB_ID}.txt

#echo "test 2475004 model on faceshifter_test"
#python -u test_open_set.py -b 1 --n_classes_total 5 --test_list data_list/ff++_fsh_test.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path cross_val_test_results/ff++_fsh/test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee cross_val_test_results/ff++_fsh/test_${SLURM_JOB_ID}.txt

#echo "test 2475004 model on faceshifter_test"
#python -u test_open_set.py -b 1 --n_classes_total 5 --test_list data_list/ff++_test.txt --resume model_save/res101/2475004/model_best.pth.tar --output_path cross_val_test_results/ff++/test_detail_${SLURM_JOB_ID}_model_2475004.txt| tee cross_val_test_results/ff++/test_${SLURM_JOB_ID}.txt
