#BSUB -o out/pdfrate_stable_0.99_brendel.%J
#BSUB -e out/pdfrate_stable_0.99_brendel.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J pdfrate_stable_0.99_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \
  -d pdfrate -i models/pdfrate_stable_0.99.h5:custom_sigmoid \
  --all -o attack_data2/pdfrate_stable_0.99_brendel.json -m pdf
