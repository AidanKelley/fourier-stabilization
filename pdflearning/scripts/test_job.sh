#BSUB -o test_job.%J
#BSUB -e test_job.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=1]"
#BSUB -J test_job

cd ~/codnn/pdflearning
pipenv run python train.py pdfrate -a tanh -o bsub_test_tmp.h5
