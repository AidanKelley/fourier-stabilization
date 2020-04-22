#BSUB -o out/mnist_train_job.%J
#BSUB -e out/mnist_train_job.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=1]"
#BSUB -J mnist_train_job

cd ~/codnn/pdflearning
pipenv run python train.py mnist -a tanh -o mnist_models/mnist_2000_epochs.h5 -c 5 -e 2000 -s tmp/mnist_2000_epochs_status.txt

