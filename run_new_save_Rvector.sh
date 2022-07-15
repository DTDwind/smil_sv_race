
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/DTDwind/.conda/envs/vox_env
python3 trainSpeakerNet.py \
        --eval \
        --model ResNetSE34L  \
        --encoder_type SAP  \
        --trainfunc angleproto  \
        --optimizer adam  \
        --save_path data/ResNetSE34L_feat_vox1_test \
        --batch_size 300  \
        --max_frames 200  \
        --nSpeakers 2  \
        --train_list data_list/vox2020Baseline/train_list.txt  \
        --train_path /mnt/HDD/HDD2/DTDwind/VoxCeleb2/aac  \
        --test_path /mnt/HDD/HDD2/DTDwind/vox1_test/wav  \
        --test_list data_list/vox2020Baseline/veri_test.txt \
        --test_interval 1 \
        --nDataLoaderThread 10 \
        --SpeakerNet_type SpeakerNet_cal_all_Dvector \
        --initial_model baseline_lite_ap.model


        # --test_path /share/corpus/voxceleb/voxsrc2020_test/voxsrc2020wav/voxsrc2020 \
        # --test_list data_list/vox2020Baseline/voxsrc2020_final_pairs.txt \