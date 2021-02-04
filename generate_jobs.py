import os

dir = "models3"

files = [
    "fraud_stable_no_acc0.96.h5",
    "fraud_stable_no_acc0.97.h5",
    "fraud_stable_no_acc0.98.h5",
    "hatespeech_stable_no_acc0.88.h5",
    "hatespeech_stable_0.89.h5",
    "hatespeech_stable_0.90.h5",
    "hidost_stable_0.996.h5",
    "hidost_stable_0.993.h5",
    "hidost_stable_0.99.h5",
    "pdfrate_stable_0.99.h5",
    "pdfrate_stable_0.985.h5",
    "pdfrate_stable_0.98.h5",
]

for filename in os.listdir(dir):

    assert(".h5" in filename)
    name = filename.split(".h5")[0]

    if "hidost" in name:
        dataset = "hidost_scaled"
    elif "pdfrate" in name:
        dataset = "pdfrate"
    elif "hatespeech" in name:
        dataset = "hatespeech"
    else:
        assert("fraud" in name)
        dataset = "fraud"

    for attack in ["custom_jsma", "brendel"]:
        out_name = f"{name}_{attack}"

        script = f"""#BSUB -o out/{out_name}.%J
#BSUB -e out/{out_name}.err.%J
#BSUB -N
#BSUB -R '(!gpu)'
#BSUB -R "rusage[mem=10]"
#BSUB -J {out_name}

cd ~/codnn
source env/bin/activate

python attack.py {attack} \\
  -d {dataset} -i {dir}/{filename}:custom_sigmoid \\
  --all -o attack_data3/{out_name}.json -m pdf
"""
        script_name = f"scripts3/{out_name}.sh"

        with open(script_name, "w") as script_out:
            script_out.write(script)


