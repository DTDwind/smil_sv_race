#!/bin/bash
cd /share/nas165/teinhonglo/AcousticModel/vox2/v4
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
ivector-plda-scoring --normalize-length=true "ivector-copy-plda --smoothing=0.0 exp/xvector_nnet_1a/xvectors_train/plda - |" "ark:ivector-subtract-global-mean exp/xvector_nnet_1a/xvectors_train/mean.vec scp:exp/xvector_nnet_1a/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec exp/xvector_nnet_1a/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" "ark:ivector-subtract-global-mean exp/xvector_nnet_1a/xvectors_train/mean.vec scp:exp/xvector_nnet_1a/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec exp/xvector_nnet_1a/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" "cat 'data/voxceleb1_test/trials' | cut -d\  --fields=1,2 |" exp/xvector_nnet_1a/xvectors_train/scores_voxceleb1_test 
EOF
) >exp/xvector_nnet_1a/xvectors_train/log/voxceleb1_test_scoring.log
time1=`date +"%s"`
 ( ivector-plda-scoring --normalize-length=true "ivector-copy-plda --smoothing=0.0 exp/xvector_nnet_1a/xvectors_train/plda - |" "ark:ivector-subtract-global-mean exp/xvector_nnet_1a/xvectors_train/mean.vec scp:exp/xvector_nnet_1a/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec exp/xvector_nnet_1a/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" "ark:ivector-subtract-global-mean exp/xvector_nnet_1a/xvectors_train/mean.vec scp:exp/xvector_nnet_1a/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec exp/xvector_nnet_1a/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" "cat 'data/voxceleb1_test/trials' | cut -d\  --fields=1,2 |" exp/xvector_nnet_1a/xvectors_train/scores_voxceleb1_test  ) 2>>exp/xvector_nnet_1a/xvectors_train/log/voxceleb1_test_scoring.log >>exp/xvector_nnet_1a/xvectors_train/log/voxceleb1_test_scoring.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/xvector_nnet_1a/xvectors_train/log/voxceleb1_test_scoring.log
echo '#' Finished at `date` with status $ret >>exp/xvector_nnet_1a/xvectors_train/log/voxceleb1_test_scoring.log
[ $ret -eq 137 ] && exit 100;
touch exp/xvector_nnet_1a/xvectors_train/q/sync/done.32703
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/xvector_nnet_1a/xvectors_train/q/voxceleb1_test_scoring.log  -l mem_free=4G,ram_free=4G   /share/nas165/teinhonglo/AcousticModel/vox2/v4/exp/xvector_nnet_1a/xvectors_train/q/voxceleb1_test_scoring.sh >>exp/xvector_nnet_1a/xvectors_train/q/voxceleb1_test_scoring.log 2>&1
