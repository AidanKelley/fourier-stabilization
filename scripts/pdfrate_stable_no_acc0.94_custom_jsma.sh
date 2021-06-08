#BSUB -o out/pdfrate_stable_no_acc0.94_custom_jsma.%J
#BSUB -e out/pdfrate_stable_no_acc0.94_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J hidost_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d pdfrate -i models/pdfrate_stable_no_acc0.94.h5:custom_sigmoid \
  --all -o attack_data/pdfrate_stable_no_acc0.94_custom_jsma.json -m pdf
