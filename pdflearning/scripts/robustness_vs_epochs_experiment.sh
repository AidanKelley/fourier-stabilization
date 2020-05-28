#BSUB -o out/$0.out.%J
#BSUB -e out/$0.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=8]"
#BSUB -J mnist_thresh_train_job

cd ~/codnn/pdflearning

pipenv run python train.py mnist_thresh -a custom_sigmoid -o mnist_thresh/sigmoid_20_epochs.h5 -c 1 -e 20 -s tmp/robustness_vs_epochs_experiment_log.txt
pipenv run python train.py mnist_thresh -i mnist_thresh/sigmoid_20_epochs.h5:custom_sigmoid -o mnist_thresh/sigmoid_40_epochs.h5 -c 1 -e 20 -s tmp/robustness_vs_epochs_experiment_log.txt
pipenv run python train.py mnist_thresh -i mnist_thresh/sigmoid_40_epochs.h5:custom_sigmoid -o mnist_thresh/sigmoid_60_epochs.h5 -c 1 -e 20 -s tmp/robustness_vs_epochs_experiment_log.txt
pipenv run python train.py mnist_thresh -i mnist_thresh/sigmoid_60_epochs.h5:custom_sigmoid -o mnist_thresh/sigmoid_80_epochs.h5 -c 1 -e 20 -s tmp/robustness_vs_epochs_experiment_log.txt
pipenv run python train.py mnist_thresh -i mnist_thresh/sigmoid_80_epochs.h5:custom_sigmoid -o mnist_thresh/sigmoid_100_epochs.h5 -c 1 -e 20 -s tmp/robustness_vs_epochs_experiment_log.txt

pipenv run python attack.py custom_jsma -n 100 \
  -o mnist_thresh/robustness_vs_epochs_experiment_log.json \
  -d mnist_thresh -i mnist_thresh/sigmoid_20_epochs.h5:custom_sigmoid \
  -d mnist_thresh -i mnist_thresh/sigmoid_40_epochs.h5:custom_sigmoid \
  -d mnist_thresh -i mnist_thresh/sigmoid_60_epochs.h5:custom_sigmoid \
  -d mnist_thresh -i mnist_thresh/sigmoid_80_epochs.h5:custom_sigmoid \
  -d mnist_thresh -i mnist_thresh/sigmoid_100_epochs.h5:custom_sigmoid \

