export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python3 trainSpeakerNet.py \
        --model ResNetSE34L  \
        --encoder_type SAP  \
        --trainfunc angleproto  \
        --optimizer adam  \
        --save_path data/feat_test \
        --batch_size 5  \
        --max_frames 200  \
        --nSpeakers 2  \
        --train_list data_list/vox2020Baseline/very_small_train_list.txt  \
        --train_path /share/nas165/chengsam/vox2/voxceleb2_dev/aac  \
        --test_path /share/nas165/chengsam/vox1/voxceleb1_test/wav  \
        --test_list data_list/vox2020Baseline/veri_test.txt \
        --test_interval 1 \
        --initial_model baseline_lite_ap.model \
        --nDataLoaderThread 10
        
# --train_list data_list/vox2020Baseline/train_list.txt  \
        # --save_path data/small_data  \
        # --train_list data_list/vox2020Baseline/train_cut.txt  \

        # --train_list data_list/vox2020Baseline/test_2.txt  \

        # --train_path /share/nas165/chengsam/vox2/voxceleb2_dev/aac  \