#BSUB -o out/mnist_train_thresh_sig_job.%J
#BSUB -e out/mnist_train_thresh_sig_job.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=8]"
#BSUB -J mnist_thresh_train_job

cd ~/codnn/pdflearning

pipenv run python train.py mnist_thresh -a custom_sigmoid -o mnist_thresh/sigmoid_500_epochs.h5 -c 5 -e 500 -s tmp/mnist_thresh_sigmoid_500_epochs_status.txt

pipenv run python stabilize.py l0 mnist_thresh mnist_thresh/sigmoid_500_epochs.h5:custom_sigmoid -o mnist_thresh/sigmoid_500_epochs_stabilized.h5

pipenv run python train.py mnist_thresh -w -i mnist_thresh/sigmoid_500_epochs_stabilized.h5:custom_sigmoid -o mnist_thresh/sigmoid_500_epochs_stabilized_retrain.h5 -c 5 -e 500 -s tmp/mnist_thresh_sigmoid_500_epochs_status.txt

pipenv run python attack.py custom_jsma -d mnist_thresh -i mnist_thresh/sigmoid_500_epochs.h5:custom_sigmoid \
  -d mnist_thresh -i mnist_thresh/sigmoid_500_epochs_stabilized.h5:custom_sigmoid \
  -d mnist_thresh -i mnist_thresh/sigmoid_500_epochs_stabilized_retrain.h5:custom_sigmoid \
  -o mnist_thresh/custom_jsma_none_stabilized_retrain_1000.json \
  -n 1000

