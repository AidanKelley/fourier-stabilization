#BSUB -o out/mnist_60_brendel_stable.%J
#BSUB -e out/mnist_60_brendel_stable.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J mnist_60_brendel_stable

cd ~/codnn/pdflearning
source env/bin/activate

python attack.py brendel -n 100 \
  -o attack_data_final/mnist_60_brendel.json \
  -d mnist_thresh -i models_final/mnist_sigmoid_60_stable_retrain.h5:custom_sigmoid
