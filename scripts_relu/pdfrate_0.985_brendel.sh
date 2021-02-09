#BSUB -o out/pdfrate_0.985_brendel.%J
#BSUB -e out/pdfrate_0.985_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J pdfrate_0.985_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d pdfrate -i models_relu/pdfrate_0.985.h5:relu \
  --all -o attack_data_relu/pdfrate_0.985_brendel.json -m pdf
