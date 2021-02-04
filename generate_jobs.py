import os

dir = "models"

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
#BSUB -J hidost_brendel

cd ~/codnn
source env/bin/activate

python attack.py brendel \\
  -d {dataset} -i {dir}/{filename}:custom_sigmoid \\
  --all -o attack_data/{out_name}.json -m pdf
"""
        script_name = f"scripts/{out_name}.sh"

        with open(script_name, "w") as script_out:
            script_out.write(script)


