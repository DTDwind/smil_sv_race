#!/bin/bash
cd /share/nas165/teinhonglo/AcousticModel/vox2/v4
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
ivector-mean ark:data/train/spk2utt scp:exp/xvector_nnet_1a/xvectors_train/xvector.scp ark,scp:exp/xvector_nnet_1a/xvectors_train/spk_xvector.ark,exp/xvector_nnet_1a/xvectors_train/spk_xvector.scp ark,t:exp/xvector_nnet_1a/xvectors_train/num_utts.ark 
EOF
) >exp/xvector_nnet_1a/xvectors_train/log/speaker_mean.log
time1=`date +"%s"`
 ( ivector-mean ark:data/train/spk2utt scp:exp/xvector_nnet_1a/xvectors_train/xvector.scp ark,scp:exp/xvector_nnet_1a/xvectors_train/spk_xvector.ark,exp/xvector_nnet_1a/xvectors_train/spk_xvector.scp ark,t:exp/xvector_nnet_1a/xvectors_train/num_utts.ark  ) 2>>exp/xvector_nnet_1a/xvectors_train/log/speaker_mean.log >>exp/xvector_nnet_1a/xvectors_train/log/speaker_mean.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/xvector_nnet_1a/xvectors_train/log/speaker_mean.log
echo '#' Finished at `date` with status $ret >>exp/xvector_nnet_1a/xvectors_train/log/speaker_mean.log
[ $ret -eq 137 ] && exit 100;
touch exp/xvector_nnet_1a/xvectors_train/q/sync/done.28360
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/xvector_nnet_1a/xvectors_train/q/speaker_mean.log -l mem_free=4G,ram_free=4G    /share/nas165/teinhonglo/AcousticModel/vox2/v4/exp/xvector_nnet_1a/xvectors_train/q/speaker_mean.sh >>exp/xvector_nnet_1a/xvectors_train/q/speaker_mean.log 2>&1
