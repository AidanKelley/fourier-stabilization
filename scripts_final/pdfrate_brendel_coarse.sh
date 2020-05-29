#BSUB -o out/pdfrate_brendel.%J
#BSUB -e out/pdfrate_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J pdfrate_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d TEST_pdfrate -i models_final_real/pdfrate_sigmoid.h5:custom_sigmoid \
  -d TEST_pdfrate -i models_final_real/pdfrate_sigmoid_stable_bias.h5:custom_sigmoid \
  --all -c 100 -o attack_data_final_real/pdfrate_brendel.json
