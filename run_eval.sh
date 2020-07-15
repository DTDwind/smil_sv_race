export CUDA_VISIBLE_DEVICES=3
nvidia-smi
python3 ./trainSpeakerNet.py \
        --eval \
        --model ResNetSE34L \
        --trainfunc angleproto  \
        --save_path data/test2  \
        --max_frames 300  \
        --test_list data_list/vox2020Baseline/veri_test.txt  \
        --test_path /share/nas165/chengsam/vox1/voxceleb1_test/wav  \
        --initial_model baseline_lite_ap.model 