#BSUB -o out/mnist_train_gray_sig_job.%J
#BSUB -e out/mnist_train_gray_sig_job.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=1]"
#BSUB -J mnist_gray_train_job

cd ~/codnn/pdflearning
pipenv run python train.py mnist_gray -a custom_sigmoid -o mnist_models/mnist_gray_sigmoid_500_epochs.h5 -c 5 -e 500 -s tmp/mnist_gray_sigmoid_500_epochs_status.txt

