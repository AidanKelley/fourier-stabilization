#BSUB -o out/mnist_train_thresh_tanh_job.%J
#BSUB -e out/mnist_train_thresh_tanh_job.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=8]"
#BSUB -J mnist_thresh_tanh_train_job

cd ~/codnn/pdflearning

pipenv run python train.py mnist_thresh -a tanh -o mnist_thresh/tanh_500_epochs.h5 -c 5 -e 500 -s tmp/mnist_thresh_tanh_500_epochs_status.txt

pipenv run python stabilize.py l0 mnist_thresh mnist_thresh/tanh_500_epochs.h5:tanh -o mnist_thresh/tanh_500_epochs_stabilized.h5

pipenv run python train.py mnist_thresh -w -i mnist_thresh/tanh_500_epochs_stabilized.h5:tanh -o mnist_thresh/tanh_500_epochs_stabilized_retrain.h5 -c 5 -e 500 -s tmp/mnist_thresh_tanh_500_epochs_status.txt

pipenv run python attack.py custom_jsma -d mnist_thresh -i mnist_thresh/tanh_500_epochs.h5:tanh \
  -d mnist_thresh -i mnist_thresh/tanh_500_epochs_stabilized.h5:tanh \
  -d mnist_thresh -i mnist_thresh/tanh_500_epochs_stabilized_retrain.h5:tanh \
  -o mnist_thresh/custom_jsma_tanh_none_stabilized_retrain_1000.json \
  -n 1000

