export CUDA_VISIBLE_DEVICES=2
nvidia-smi
python3.6 trainSpeakerNet.py \
        --model ResNetSE34L  \
        --encoder_type SAP  \
        --trainfunc triplet  \
        --optimizer adam  \
        --save_path data/triplet2 \
        --batch_size 500  \
        --max_frames 200  \
        --nSpeakers 2  \
        --train_list data_list/vox2020Baseline/train_list.txt  \
        --train_path /share/nas165/chengsam/vox2/voxceleb2_dev/aac  \
        --test_path /share/nas165/chengsam/vox1/voxceleb1_test/wav  \
        --test_list data_list/vox2020Baseline/veri_test.txt \
        --test_interval 1 \
        --margin 0.1 \
        --lr_decay 0.95 \
        --initial_model initial_model/triplet0814.model \
        --nDataLoaderThread 10 \
        --SpeakerNet_type SpeakerNet_triplet
