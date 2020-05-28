#BSUB -o out/mnist_60_brendel_orig.%J
#BSUB -e out/mnist_60_brendel_orig.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J mnist_60_brendel_orig

cd ~/codnn/pdflearning
source env/bin/activate

python attack.py brendel --all \
  -o attack_data_final/mnist_60_brendel.json \
  -d mnist_thresh -i models_final/mnist_sigmoid_60.h5:custom_sigmoid 
