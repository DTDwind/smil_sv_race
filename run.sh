export CUDA_VISIBLE_DEVICES=3
nvidia-smi
python3 trainSpeakerNet.py \
        --model ResNetSE34L  \
        --encoder_type SAP  \
        --trainfunc angleproto  \
        --optimizer adam  \
        --save_path data/lstm_big_test \
        --batch_size 400  \
        --max_frames 200  \
        --nSpeakers 2  \
        --train_list data_list/vox2020Baseline/train_list.txt  \
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

        # --save_path data/lstm_test \ # 這個是用小筆的... train_cut.txt