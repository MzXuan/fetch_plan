#!/bin/bash

trap finish SIGINT

function finish() {
        echo "bye bye!"
        exit
}

rl_model="./log"
pred_model="./pred"


counter=0
save_counter=0
while [ ${counter} -le ${1} ]
do
echo $counter

# train rl
if [ ${counter} -eq 0 ]
then
    python run.py -t=True --pred_weight=0.0
else
    python run.py -t=True -l=True -p='00200' --pred_weight=${2}
fi

# copy saved file and rename
cp -R ${rl_model} "./models/log_${counter}"
cp -R ${pred_model} "./models/pred_${counter}"

# run new training cycle
sleep 1

# sample dataset
python run.py -l=True -p='00200'
sleep 1

# train seq2seq
python predictor.py -l=True --iter=${counter}

((counter++))

done

cp -R ${rl_model} "./models/log_${counter}"
cp -R ${pred_model} "./models/pred_${counter}"
echo All done
