#BSUB -o out/mnist_train_bin_tanh_job.%J
#BSUB -e out/mnist_train_bin_tanh_job.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=1]"
#BSUB -J mnist_bin_tanh_train_job

cd ~/codnn/pdflearning
pipenv run python train.py mnist_bin -a tanh -o mnist_models/mnist_bin_tanh_500_epochs.h5 -c 5 -e 500 -s tmp/mnist_bin_tanh_500_epochs_status.txt

