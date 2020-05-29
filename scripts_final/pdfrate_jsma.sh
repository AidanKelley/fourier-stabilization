#BSUB -o out/pdfrate_jsma.%J
#BSUB -e out/pdfrate_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J pdfrate_jsma

cd ~/codnn
source env/bin/activate

python attack.py custom_jsma --all \
  -o attack_data_final_real/pdfrate_jsma.json \
  -d TEST_pdfrate -i models_final_real/pdfrate_sigmoid.h5:custom_sigmoid \
  -d TEST_pdfrate -i models_final_real/pdfrate_sigmoid_stable_bias.h5:custom_sigmoid
