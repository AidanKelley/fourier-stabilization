#BSUB -o out/mnist_thresh_similarity_job.%J
#BSUB -e out/mnist_thresh_similarity_job.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J mnist_thresh_similarity_job

cd ~/codnn/pdflearning
source env/bin/activate
python weight_similarity.py mnist_thresh -i models_final/mnist_sigmoid_stable_retrain.h5:custom_sigmoid -o similarity_out_final/mnist_similarity.json -N 100000 -N 200000 -N 400000 -N 800000

