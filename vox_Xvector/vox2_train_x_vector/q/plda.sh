#!/bin/bash
cd /share/nas165/teinhonglo/AcousticModel/vox2/v4
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
ivector-compute-plda ark:data/train/spk2utt "ark:ivector-subtract-global-mean scp:exp/xvector_nnet_1a/xvectors_train/xvector.scp ark:- | transform-vec exp/xvector_nnet_1a/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" exp/xvector_nnet_1a/xvectors_train/plda 
EOF
) >exp/xvector_nnet_1a/xvectors_train/log/plda.log
time1=`date +"%s"`
 ( ivector-compute-plda ark:data/train/spk2utt "ark:ivector-subtract-global-mean scp:exp/xvector_nnet_1a/xvectors_train/xvector.scp ark:- | transform-vec exp/xvector_nnet_1a/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" exp/xvector_nnet_1a/xvectors_train/plda  ) 2>>exp/xvector_nnet_1a/xvectors_train/log/plda.log >>exp/xvector_nnet_1a/xvectors_train/log/plda.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/xvector_nnet_1a/xvectors_train/log/plda.log
echo '#' Finished at `date` with status $ret >>exp/xvector_nnet_1a/xvectors_train/log/plda.log
[ $ret -eq 137 ] && exit 100;
touch exp/xvector_nnet_1a/xvectors_train/q/sync/done.32102
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/xvector_nnet_1a/xvectors_train/q/plda.log -l mem_free=4G,ram_free=4G    /share/nas165/teinhonglo/AcousticModel/vox2/v4/exp/xvector_nnet_1a/xvectors_train/q/plda.sh >>exp/xvector_nnet_1a/xvectors_train/q/plda.log 2>&1
