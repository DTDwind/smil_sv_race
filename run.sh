export CUDA_VISIBLE_DEVICES=3
python3 trainSpeakerNet.py \
        --model ResNetSE34L  \
        --encoder_type SAP  \
        --trainfunc angleproto  \
        --optimizer adam  \
        --save_path data/loss_test2  \
        --batch_size 100  \
        --max_frames 200  \
        --nSpeakers 2  \
        --initial_model /share/nas165/Wendy/smil_sv_race/models/baseline_lite_ap.model \
        --train_list /share/nas165/Wendy/smil_sv_race/data_list/test_2.txt \
        --train_path /share/nas165/chengsam/vox2/voxceleb2_dev/aac  \
        --test_path /share/nas165/chengsam/vox1/voxceleb1_test/wav  \
        --test_list data_list/vox2020Baseline/veri_test.txt \
        --test_interval 1 \
        --nDataLoaderThread 10 \
#python3 ./trainSpeakerNet.py --eval --model ResNetSE34L --trainfunc angleproto --save_path data/test --max_frames 300 --test_list data_list/vox2020Baseline/veri_test.txt --test_path dataset/voxceleb1_test/wav --initial_model /share/nas165/Wendy/smil_sv_race/models/baseline_lite_ap.model
#/share/nas165/Wendy/smil_sv_race/data_list/test_2.txt
#/share/nas165/Wendy/smil_sv_race/data_list/train_list.txt 




