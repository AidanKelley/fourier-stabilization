#BSUB -o out/mnist_60_brendel_coarse.%J
#BSUB -e out/mnist_60_brendel_coarse.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J mnist_60_brendel_coarse

cd ~/codnn/pdflearning
source env/bin/activate

python attack.py brendel --all -c 100 \
  -o attack_data_final_real/mnist_60_brendel_coarse.json \
  -d TEST_mnist_thresh -i models_final/mnist_sigmoid_60.h5:custom_sigmoid \
  -d TEST_mnist_thresh -i models_final/mnist_sigmoid_60_stable_retrain.h5:custom_sigmoid
