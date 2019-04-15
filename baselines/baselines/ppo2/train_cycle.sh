#!/bin/bash

trap finish SIGINT

function finish() {
        echo "bye bye!"
        exit
}

# preparation
rm -rf "./models"
mkdir "./models"

rl_model="./log/"
pred_model="./pred/"


# start training
counter=0
while [ ${counter} -le ${1} ]
do
echo $counter

# train rl
if [ ${counter} -eq 0 ]
then
    python run.py --train --num-timesteps=1300000 --pred_weight=0.0 --iter=${counter}
else
    python run.py --train --load --num-timesteps=1300000 -p='00150' --pred_weight=${2} --iter=${counter}
fi

# run new training cycle
sleep 1

# sample dataset
python run.py --load -p='last'

sleep 1

# train seq2seq
if [ ${counter} -eq 0 ]
then
    python predictor.py --iter=${counter} --lr=0.001
else
    python predictor.py --load --iter=${counter} --lr=0.0001
fi

# copy saved file and rename
cp -R ${rl_model} "./models/log_${counter}"
cp -R ${pred_model} "./models/pred_${counter}"

rm -rf "${pred_model}/test1/checkpoint_" 
rm -rf "${rl_model}/tb"

((counter++))
done

echo All done