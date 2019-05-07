#!/bin/bash

trap finish SIGINT

function finish() {
        echo "bye bye!"
        exit
}

# preparation
rl_model="./log/"
pred_model="./pred/"

# start from beginning
rm -rf "./models"
mkdir "./models"
counter=0

# # start with trained initial model
# counter=1


# start training
while [ ${counter} -le ${1} ]
do
echo $counter

# train rl
if [ ${counter} -eq 0 ]
then
    python run.py --train --num-timesteps=10000000 --pred_weight=0.0 --iter=${counter}
else
    python run.py --train --load --num-timesteps=1000000 -p='last' --pred_weight=${2} --iter=${counter}
fi

# run new training cycle
sleep 1

# sample dataset
if [ ${counter} -eq 0 ]
then
    python run.py --load -p='last'
else
    python run.py --load -p='last'
fi

sleep 1

# train seq2seq
if [ ${counter} -eq 0 ]
then
    python predictor.py --iter=${counter} --lr=0.03 --epoch=20
    sleep 1
    python predictor.py --load --iter=${counter} --lr=0.00025 --epoch=300
elif [ ${counter} -le 3 ]
then
    python predictor.py --load --iter=${counter} --lr=0.0003 --epoch=300
else
    python predictor.py --load --iter=${counter} --lr=0.0001 --epoch=200
fi

# copy saved file and rename
cp -R ${rl_model} "./models/log_${counter}"
cp -R ${pred_model} "./models/pred_${counter}"

rm -rf "${pred_model}/test1/checkpoint_" 
rm -rf "${rl_model}/tb"

((counter++))
done

echo All done
