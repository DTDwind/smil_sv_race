export CUDA_VISIBLE_DEVICES=2
# nvidia-smi
time python3.6 ./trainSpeakerNet.py \
        --eval \
        --model ResNetSE34L \
        --trainfunc angleproto  \
        --save_path data/eat_pairwise_distance2  \
        --max_frames 200  \
        --test_list data_list/vox2020Baseline/very_small_32_eal_test.txt  \
        --test_path /share/nas165/chengsam/vox1/voxceleb1_test/wav  \
        --SpeakerNet_type SpeakerNet_eat_pairwise_distance