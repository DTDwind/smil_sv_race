# 以後用這個當測試專用的...
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /share/homes/chengsam/miniconda3/envs/vox_env
python3 trainSpeakerNet.py \
        --model ResNetSE34L  \
        --encoder_type SAP  \
        --trainfunc angleproto  \
        --optimizer adam  \
        --save_path data/QAQ_test \
        --batch_size 3  \
        --max_frames 200  \
        --nSpeakers 2  \
        --train_list data_list/vox2020Baseline/test_2.txt  \
        --train_path /share/corpus/voxceleb/vox2_trainset/aac/aac  \
        --test_path /share/nas165/chengsam/vox1/voxceleb1_test/wav  \
        --test_list data_list/vox2020Baseline/very_small_32_eal_test.txt \
        --test_interval 1 \
        --initial_model baseline_lite_ap.model \
        --nDataLoaderThread 1 \
        --SpeakerNet_type SpeakerNet_QAQ