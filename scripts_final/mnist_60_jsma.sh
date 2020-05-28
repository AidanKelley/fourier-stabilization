#BSUB -o out/mnist_60_jsma.%J
#BSUB -e out/mnist_60_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J mnist_60_jsma

cd ~/codnn/pdflearning
source env/bin/activate

python attack.py custom_jsma --all \
  -o attack_data_final_real/mnist_60_jsma.json \
  -d TEST_mnist_thresh -i models_final/mnist_sigmoid_60.h5:custom_sigmoid \
  -d TEST_mnist_thresh -i models_final/mnist_sigmoid_60_stable_retrain.h5:custom_sigmoid
