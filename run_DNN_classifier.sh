export CUDA_VISIBLE_DEVICES=1
nvidia-smi
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /share/homes/chengsam/miniconda3/envs/vox_env
python3 trainSpeakerNet.py \
        --model ResNetSE34L  \
        --encoder_type SAP  \
        --trainfunc angleproto  \
        --optimizer adam  \
        --save_path data/DNN_classifier3 \
        --batch_size 400  \
        --max_frames 200  \
        --nSpeakers 2  \
        --train_list data_list/vox2020Baseline/train_list.txt  \
        --train_path /share/nas165/chengsam/vox2/voxceleb2_dev/aac  \
        --test_path /share/nas165/chengsam/vox1/voxceleb1_test/wav  \
        --test_list data_list/vox2020Baseline/veri_test.txt \
        --test_interval 1 \
        --initial_model baseline_lite_ap.model \
        --nDataLoaderThread 3 \
        --SpeakerNet_type SpeakerNet_DNN_classifier
