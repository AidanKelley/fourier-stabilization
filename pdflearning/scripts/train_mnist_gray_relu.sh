#BSUB -o out/mnist_train_gray_relu_job.%J
#BSUB -e out/mnist_train_gray_relu_job.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=1]"
#BSUB -J mnist_relu_train_job

cd ~/codnn/pdflearning
pipenv run python train.py mnist_gray -a relu -o mnist_models/mnist_gray_relu_500_epochs.h5 -c 5 -e 500 -s tmp/mnist_gray_relu_500_epochs_status.txt

