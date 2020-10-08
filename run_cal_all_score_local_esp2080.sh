export CUDA_VISIBLE_DEVICES=1
nvidia-smi
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /share/homes/chengsam/miniconda3/envs/vox_env
python3 trainSpeakerNet.py \
        --eval \
        --model ResNetSE34L  \
        --encoder_type SAP  \
        --trainfunc angleproto  \
        --optimizer adam  \
        --save_path data/all_score_local \
        --batch_size 300  \
        --max_frames 200  \
        --nSpeakers 2  \
        --train_list data_list/vox2020Baseline/train_cut.txt  \
        --train_path /share/nas165/chengsam/vox2/voxceleb2_dev/aac  \
        --test_path /share/corpus/voxceleb/voxsrc2020_test/voxsrc2020wav/voxsrc2020 \
        --test_list data_list/vox2020Baseline/voxsrc2020_final_pairs.txt \
        --test_interval 1 \
        --nDataLoaderThread 10 \
        --SpeakerNet_type SpeakerNet_cal_all_score_loacl_esp2080  \
        --initial_model baseline_lite_ap.model


        # --test_path /share/corpus/voxceleb/voxsrc2020_test/voxsrc2020wav/voxsrc2020 \
        # --test_list data_list/vox2020Baseline/voxsrc2020_final_pairs.txt \

        # --test_list data_list/vox2020Baseline/veri_test.txt  \
        # --test_path /share/nas165/chengsam/vox1/voxceleb1_test/wav  \
        
