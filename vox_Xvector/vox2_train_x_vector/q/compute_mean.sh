#!/bin/bash
cd /share/nas165/teinhonglo/AcousticModel/vox2/v4
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
ivector-mean scp:exp/xvector_nnet_1a/xvectors_train/xvector.scp exp/xvector_nnet_1a/xvectors_train/mean.vec 
EOF
) >exp/xvector_nnet_1a/xvectors_train/log/compute_mean.log
time1=`date +"%s"`
 ( ivector-mean scp:exp/xvector_nnet_1a/xvectors_train/xvector.scp exp/xvector_nnet_1a/xvectors_train/mean.vec  ) 2>>exp/xvector_nnet_1a/xvectors_train/log/compute_mean.log >>exp/xvector_nnet_1a/xvectors_train/log/compute_mean.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/xvector_nnet_1a/xvectors_train/log/compute_mean.log
echo '#' Finished at `date` with status $ret >>exp/xvector_nnet_1a/xvectors_train/log/compute_mean.log
[ $ret -eq 137 ] && exit 100;
touch exp/xvector_nnet_1a/xvectors_train/q/sync/done.32027
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/xvector_nnet_1a/xvectors_train/q/compute_mean.log  -l mem_free=4G,ram_free=4G   /share/nas165/teinhonglo/AcousticModel/vox2/v4/exp/xvector_nnet_1a/xvectors_train/q/compute_mean.sh >>exp/xvector_nnet_1a/xvectors_train/q/compute_mean.log 2>&1
