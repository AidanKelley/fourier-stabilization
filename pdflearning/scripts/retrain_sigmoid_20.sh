#BSUB -o out/mnist_retrain_thresh_sig_job.%J
#BSUB -e out/mnist_retrain_thresh_sig_job.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=8]"
#BSUB -J mnist_thresh_train_job

cd ~/codnn/pdflearning

pipenv run python stabilize.py l0 mnist_thresh mnist_thresh/sigmoid_20_epochs.h5:custom_sigmoid -o mnist_thresh/sigmoid_20_epochs_stabilized.h5

pipenv run python train.py mnist_thresh -w -i mnist_thresh/sigmoid_20_epochs_stabilized.h5:custom_sigmoid -o mnist_thresh/sigmoid_20_epochs_stabilized_retrain.h5 -c 1 -e 100 -s tmp/mnist_thresh_retrain_sigmoid_20_epochs_status.txt
