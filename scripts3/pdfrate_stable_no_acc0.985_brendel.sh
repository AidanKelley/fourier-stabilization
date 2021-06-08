#BSUB -o out/pdfrate_stable_no_acc0.985_brendel.%J
#BSUB -e out/pdfrate_stable_no_acc0.985_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J pdfrate_stable_no_acc0.985_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d pdfrate -i models3/pdfrate_stable_no_acc0.985.h5:custom_sigmoid \
  --all -o attack_data3/pdfrate_stable_no_acc0.985_brendel.json -m pdf
