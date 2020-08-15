export CUDA_VISIBLE_DEVICES=2
nvidia-smi
python3.6 trainSpeakerNet.py \
        --model ResNetSE34L  \
        --encoder_type SAP  \
        --trainfunc triplet  \
        --optimizer adam  \
        --save_path data/loss_testX \
        --batch_size 32  \
        --max_frames 200  \
        --nSpeakers 2  \
        --train_list data_list/vox2020Baseline/test_2.txt  \
        --train_path /share/nas165/chengsam/vox2/voxceleb2_dev/aac  \
        --test_path /share/nas165/chengsam/vox1/voxceleb1_test/wav  \
        --test_list data_list/vox2020Baseline/veri_test.txt \
        --test_interval 1 \
        --initial_model lr_test_model.model \
        --nDataLoaderThread 10 \
        --SpeakerNet_type SpeakerNet_lr_test