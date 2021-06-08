#BSUB -o out/pdfrate_custom_jsma.%J
#BSUB -e out/pdfrate_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d pdfrate -i models/pdfrate.h5:custom_sigmoid \
  --all -o attack_data/pdfrate_custom_jsma.json -m pdf
