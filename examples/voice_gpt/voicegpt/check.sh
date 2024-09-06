#/bin/sh
while true
do
if ps ax | grep -v grep | grep "llama_finetune" > /dev/null
then
    echo "waiting for finish"
    sleep 1800
else
    echo "Process is not running"
    #./yi-34b-100k2.sh & 
    #./yi-34b-100k-autotok-noboseos.sh &
    export PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    #./yi-34b-100k-autotok-noboseos-lr-small2.sh &
    #./yi-34b-v11-ebook-100k-autotok-noboseos-lr-small.sh &
    #./yi-34b-v15-ebook-100k-test-v2.sh &
    #./sft_QWen1.5_wp10_gbs64.sh &
    #./run-13b-16k.sh &
    ./run-stage1.sh &
    break
fi
done
