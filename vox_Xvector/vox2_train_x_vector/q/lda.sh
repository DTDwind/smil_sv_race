#!/bin/bash
cd /share/nas165/teinhonglo/AcousticModel/vox2/v4
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
ivector-compute-lda --total-covariance-factor=0.0 --dim=200 "ark:ivector-subtract-global-mean scp:exp/xvector_nnet_1a/xvectors_train/xvector.scp ark:- |" ark:data/train/utt2spk exp/xvector_nnet_1a/xvectors_train/transform.mat 
EOF
) >exp/xvector_nnet_1a/xvectors_train/log/lda.log
time1=`date +"%s"`
 ( ivector-compute-lda --total-covariance-factor=0.0 --dim=200 "ark:ivector-subtract-global-mean scp:exp/xvector_nnet_1a/xvectors_train/xvector.scp ark:- |" ark:data/train/utt2spk exp/xvector_nnet_1a/xvectors_train/transform.mat  ) 2>>exp/xvector_nnet_1a/xvectors_train/log/lda.log >>exp/xvector_nnet_1a/xvectors_train/log/lda.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/xvector_nnet_1a/xvectors_train/log/lda.log
echo '#' Finished at `date` with status $ret >>exp/xvector_nnet_1a/xvectors_train/log/lda.log
[ $ret -eq 137 ] && exit 100;
touch exp/xvector_nnet_1a/xvectors_train/q/sync/done.32047
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/xvector_nnet_1a/xvectors_train/q/lda.log  -l mem_free=4G,ram_free=4G   /share/nas165/teinhonglo/AcousticModel/vox2/v4/exp/xvector_nnet_1a/xvectors_train/q/lda.sh >>exp/xvector_nnet_1a/xvectors_train/q/lda.log 2>&1
