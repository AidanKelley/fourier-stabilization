#BSUB -o out/pdfrate_custom_jsma.%J
#BSUB -e out/pdfrate_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J pdfrate_custom_jsma

cd ~/codnn
source env/bin/activate

python attack.py custom_jsma \
  -d pdfrate -i models_relu/pdfrate.h5:relu \
  --all -o attack_data_relu/pdfrate_custom_jsma.json -m pdf
