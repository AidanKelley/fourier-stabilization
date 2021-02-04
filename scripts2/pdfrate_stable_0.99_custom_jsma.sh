#BSUB -o out/pdfrate_stable_0.99_custom_jsma.%J
#BSUB -e out/pdfrate_stable_0.99_custom_jsma.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J pdfrate_stable_0.99_custom_jsma

cd ~/codnn
source env/bin/activate

python attack.py custom_jsma \
  -d pdfrate -i models/pdfrate_stable_0.99.h5:custom_sigmoid \
  --all -o attack_data2/pdfrate_stable_0.99_custom_jsma.json -m pdf
