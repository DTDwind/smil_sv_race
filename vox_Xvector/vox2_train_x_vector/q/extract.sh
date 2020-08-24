#!/bin/bash
cd /share/nas165/teinhonglo/AcousticModel/vox2/v4
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
nnet3-xvector-compute --use-gpu=no --min-chunk-size=25 --chunk-size=10000 --cache-capacity=64 "nnet3-copy --nnet-config=exp/xvector_nnet_1a/extract.config exp/xvector_nnet_1a/final.raw - |" "ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:data/train/split80/${SGE_TASK_ID}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:data/train/split80/${SGE_TASK_ID}/vad.scp ark:- |" ark,scp:exp/xvector_nnet_1a/xvectors_train/xvector.${SGE_TASK_ID}.ark,exp/xvector_nnet_1a/xvectors_train/xvector.${SGE_TASK_ID}.scp 
EOF
) >exp/xvector_nnet_1a/xvectors_train/log/extract.$SGE_TASK_ID.log
time1=`date +"%s"`
 ( nnet3-xvector-compute --use-gpu=no --min-chunk-size=25 --chunk-size=10000 --cache-capacity=64 "nnet3-copy --nnet-config=exp/xvector_nnet_1a/extract.config exp/xvector_nnet_1a/final.raw - |" "ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:data/train/split80/${SGE_TASK_ID}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:data/train/split80/${SGE_TASK_ID}/vad.scp ark:- |" ark,scp:exp/xvector_nnet_1a/xvectors_train/xvector.${SGE_TASK_ID}.ark,exp/xvector_nnet_1a/xvectors_train/xvector.${SGE_TASK_ID}.scp  ) 2>>exp/xvector_nnet_1a/xvectors_train/log/extract.$SGE_TASK_ID.log >>exp/xvector_nnet_1a/xvectors_train/log/extract.$SGE_TASK_ID.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/xvector_nnet_1a/xvectors_train/log/extract.$SGE_TASK_ID.log
echo '#' Finished at `date` with status $ret >>exp/xvector_nnet_1a/xvectors_train/log/extract.$SGE_TASK_ID.log
[ $ret -eq 137 ] && exit 100;
touch exp/xvector_nnet_1a/xvectors_train/q/sync/done.25023.$SGE_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/xvector_nnet_1a/xvectors_train/q/extract.log -l mem_free=4G,ram_free=4G   -t 1:80 /share/nas165/teinhonglo/AcousticModel/vox2/v4/exp/xvector_nnet_1a/xvectors_train/q/extract.sh >>exp/xvector_nnet_1a/xvectors_train/q/extract.log 2>&1
