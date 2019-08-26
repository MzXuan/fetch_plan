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

counter=${3}
if [ ${counter} -eq 0 ]
then
    rm -rf "./models"
    mkdir "./models"
fi


# parameter 1. totle iterations; 2. prediction weight; 3. start counter counts

# start training
while [ ${counter} -le ${1} ]
do
echo $counter

# train rl
if [ ${counter} -eq 0 ]
then
    python run.py --train --num-timesteps=5000000 --pred_weight=0.0 --iter=${counter}
else
    python run.py --train --load --num-timesteps=5000000 -p='last' --pred_weight=${2} --iter=${counter}
fi

# run new training cycle
sleep 1
cp -R ${rl_model} "./models/log_${counter}"

# sample dataset	## sam?ed_model}/test1/checkpoint_"
python run.py --load -p='last'
sleep 1


 # train seq2seq
if [ ${counter} -eq 0 ]
then
    python predictors.py --iter=${counter} --epoch=20
elif [ ${counter} -le 3 ]
then
    python predictors.py --load --iter=${counter} --epoch=20
else
    python predictors.py --load --iter=${counter} --epoch=20
fi


# copy saved file and rename
cp -R ${pred_model} "./models/pred_${counter}"	#rl
rm -rf "${pred_model}/test1/checkpoint_"

#rl
rm -rf "${rl_model}/tb"


((counter++))
done

echo All done
