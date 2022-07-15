export CUDA_VISIBLE_DEVICES=0
nvidia-smi
time python3.6 ./trainSpeakerNet.py \
        --eval \
        --model ResNetSE34L \
        --trainfunc angleproto  \
        --save_path data/ResNetSE34L_feat_vox1_test  \
        --max_frames 400  \
        --test_list data_list/vox2020Baseline/veri_test.txt  \
        --test_path /mnt/HDD/HDD2/DTDwind/vox1_test/wav  \
        --initial_model baseline_lite_ap.model 


        # EER 2.3701