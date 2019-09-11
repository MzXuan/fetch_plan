#!/bin/bash

trap finish SIGINT

function finish() {
        echo "bye bye!"
        exit
}

# parameter 1. start counter; 2. prediction weight;


# start training
counter=${1}
echo $counter

# preparation
rl_model="./log/"
pred_model="./pred/"

if [ ${counter} -eq 0 ]
then
    rm -rf "./models"
    mkdir "./models"
    mkdir "./pred"
fi

# train rl
if [ ${counter} -eq 0 ]
then
#    sleep 1
    python run.py --train --num-timesteps=5000000 --pred_weight=${2} --iter=${counter}
else
    python run.py --train --load --num-timesteps=10000000 -p='last' --pred_weight=${2} --iter=${counter}
fi

# run new training cycle
sleep 1

cp -R ${rl_model} "./models/log_${counter}"

# sample dataset
python run.py --load --seed=$((100+counter)) -p='last'


sleep 1


# train seq2seq
if [ ${counter} -eq 0 ]
then
    python predictors.py --iter=${counter} --epoch=30
elif [ ${counter} -le 3 ]
then
    python predictors.py  --iter=${counter} --epoch=30
else
    python predictors.py  --iter=${counter} --epoch=30
fi


# copy saved file and rename
cp -R ${pred_model} "./models/pred_${counter}"	#rl
rm -rf "${pred_model}/test1/checkpoint_"

#rl
rm -rf "${rl_model}/tb"


echo All done
