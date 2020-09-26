# 以後用這個當測試專用的...
export CUDA_VISIBLE_DEVICES=3
nvidia-smi
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /share/homes/chengsam/miniconda3/envs/vox_env
python3 trainSpeakerNet.py \
        --eval \
        --model ResNetSE34L  \
        --encoder_type SAP  \
        --trainfunc angleproto  \
        --optimizer adam  \
        --save_path data/QAQ \
        --batch_size 300  \
        --max_frames 200  \
        --nSpeakers 2  \
        --train_list data_list/vox2020Baseline/train_cut.txt  \
        --train_path /share/nas165/chengsam/vox2/voxceleb2_dev/aac  \
        --test_path /share/nas165/chengsam/vox1/voxceleb1_test/wav  \
        --test_list data_list/vox2020Baseline/veri_test.txt \
        --test_interval 1 \
        --nDataLoaderThread 10 \
        --SpeakerNet_type SpeakerNet_Xvector
        # --initial_model baseline_lite_ap.model \