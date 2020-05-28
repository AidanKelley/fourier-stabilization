#BSUB -o out/hidost_brendel.%J
#BSUB -e out/hidost_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d TEST_hidost_scaled -i models_final_real/hidost_sigmoid.h5:custom_sigmoid \
  -d TEST_hidost_scaled -i models_final_real/hidost_sigmoid_stable_bias.h5:custom_sigmoid \
  --all -o attack_data_final_real/hidost_brendel.json
