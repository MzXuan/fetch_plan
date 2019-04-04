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
save_counter=0
while [ ${counter} -le ${1} ]
do
echo $counter

# train rl
if [ ${counter} -eq 0 ]
then
    python run.py -t=True --num-timesteps=1860000 --pred_weight=0.0
else
    python run.py -t=True -l=True --num-timesteps=1860000 -p='00200' --pred_weight=${2}
fi


# run new training cycle
sleep 1

# sample dataset
python run.py -l=True -p='00200'
sleep 1

# train seq2seq
python predictor.py -l=True --iter=${counter}


# copy saved file and rename
cp -R ${rl_model} "./models/log_${counter}"
cp -R ${pred_model} "./models/pred_${counter}"

rm -rf "${pred_model}/test1/checkpoint_" 

((counter++))
done

echo All done