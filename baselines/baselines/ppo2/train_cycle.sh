#!/bin/bash

trap finish SIGINT

function finish() {
        echo "bye bye!"
        exit
}

rl_model="./log"
pred_model="./pred"


counter=1
while [ $counter -le 10 ]
do
echo $counter
# copy saved file and rename

cp -R ${rl_model} "./models/log_${counter}"
cp -R ${pred_model} "./models/pred_${counter}"

# run new training cycle
if [ ${counter} -eq 1 ]
then
	python run.py -t=True -l=True -p='00350'
else
	python run.py -t=True -l=True -p='00200'
fi
sleep 1
python run.py -l=True -p='00200'
sleep 1
python predictor.py

((counter++))


done
echo All done