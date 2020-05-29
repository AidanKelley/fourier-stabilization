# Welcome!
Here, we'll explain how to get the testing environment setup and running the experiments that we used to generate the graphs in our paper.

These experiments were run on a research server running linux using Intel Xeon Gold 6150 CPU's running at 2.70GHz, with the shell bash.
This codebase is not guaranteed to run on
any other machine, and in fact tensorflow is known to fail on some architectures (if the python script terminates with "Illegal Exception", this
is most likely why").

# Environment Setup
We used Python 3.7.7, 64-bit, and pip 19.2.3, running on a Linux system. It is very likely this code will not work on a Windows system.
Any version of python above 3.6 _should_ work, but has not been tested. You may need to install these, and
additionally, `venv`, but instructions for that are out of scope.
To setup the virtual env run

`python -m venv env`

Creating the env might take some time. Then, run

`source env/bin/activate`

to activate the virtual environment. Then, you'll want to install the requirements using

`pip install -r requirements.txt`

# Training models
To train a model, do

`mkdir pdfrate_sigmoid`

`python train.py pdfrate -a scaled_sigmoid -e 100 -c 1 -o pdfrate_sigmoid/{e}_epochs.h5 -s pdfrate_sigmoid/status_file.txt`

This will train a model to classify the `PDFRate-B` dataset using the scaled sigmoid activation function (described in section 5 as being a scaled logistic).
The model will be trained for 100 epochs and will be saved after every 1 epoch. It will be saved to the file `pdfrate_sigmoid_model/{e}_epochs.h5`, where {e}
is the number of epochs the model has been trained for. The validation accuracy and loss will be stored to `pdfrate_sigmoid_model/status_file.txt` every
time the model is saved.

Once the training is done, we want to pick the number of epochs that had the best validation error. This will be outputted to the console and is
also recorded in the status file. Once you find the number, make a directory where we'll store our models and move it into there

`mkdir models_final_real`

`mv pdfrate_sigmoid/{e}_epochs.h5 models_final_real/pdfrate_sigmoid.h5`

where `{e}` is the number of epochs that gave the best validation accuracy.

Now, we want to apply the l1 stabilization process to the model. Run

`python stabilize.py l1 pdfrate models_final_real/pdfrate_sigmoid.h5:scaled_sigmoid -o models_final_real/pdfrate_sigmoid_stable.h5`

This specifices that we want to do the l1 stabilization process on a model that classifies the pdfrate dataset. It says that the model we want to stabilize is stored at
`models_final_real/pdfrate_sigmoid.h5`, and the colon specifies that we are using the `scaled_sigmoid` activation function. This is necessary since the `.h5` file
does not store this information, and because our code supports multiple activation functions. Finally, we specify that the stabilized model should be saved to
`models_final_real/pdfrate_sigmoid_stable.h5`.

The stabilization process sets biases to 0, but we now want to find the optimal biases through training. Then, run

`mkdir pdfrate_sigmoid_stable_bias`

`python train.py pdfrate -i models_final_real/pdfrate_sigmoid_stable.h5:scaled_sigmoid -e 200 -c 1 -b -o pdfrate_sigmoid_stable_bias/{e}_epochs.h5 -s pdfrate_sigmoid_stable_bias/status_file`

This is similar to the first time we ran `train.py`, except this time, we passed in an input model with the `-i` option. Additionally, we do not pass in an activation
function using `-a` this time, since the activation function is passned in via a colon after the input model name. We train for 200 epochs and save after every epoch. The `-b` option means that all weights are frozen except for the biases, so it is only the biases in the hidden layer that are trained. `-o` and `-s` do the same as described previously.

Once the training has finished, we again selected the best model by validation, running

`mv pdfrate_sigmoid_stable_bias/{e}_epochs.h5 models_final_real/pdfrate_sigmoid_stable_bias.h5`

Now, we have trained the models used in the experiments. This can be repeated for the `hidost` or `mnist` datasets by substituting `pdfrate` for
`hidost_scaled` or `mnist_thresh`, respectively, as the dataset argument.
You'll also want to name your models something else to differeniate between the two datasets.

# Running attacks

You'll want to create a directory for the attack data, so run

`mkdir attack_data_final_real`

Then, we'll use the `attack.py` script to run attacks to compare the two models you've just trained, run

`python attack.py custom_jsma -d TEST_pdfrate -i models_final_real/pdfrate_sigmoid.h5:scaled_sigmoid \
     
     -d TEST_pdfrate -i models_final_real/pdfrate_sigmoid_stable_bias.h5:scaled_sigmoid
     
     --all -o attack_data_final_real/pdfrate_jsma.json` 

The first argument specifies the attack we will use, the `custom_jsma` attack. Then, we pass in a dataset and model pair. Note that you must supply
a dataset for *every* model, as the attack is designed to support comparing attacks on two models where the underlying datasets describe the same data
but with different features. We appended `TEST_` to `pdfrate` to specify that the attack will run on the true testing partition.
Without this, it would run on the validation partition. Finally, we use the `--all` option to specify that we should run the attack on every data point.
We could run it an a random subset of size `n` by supplying `-n {n}` instead. The output is saved in `json` format to `attack_data_final_real/pdfrate_jsma.json`

To run attack using the Brendel and Bethge L1 attack, simply replace `custom_jsma` with `brendel`, and of course rename the output file so your data is not overwritten.

To run the attack on models sunig `hidost` or `mnist`, simply replace both occurences of `TEST_pdfrate` with `TEST_hidost_scaled` or `TEST_mnist_thresh`, respectively.

# Viewing the output of the attack

In order to graphically display the output of the attack, run

`python plot_float_robustness attack_data_final_real/pdfrate_jsma.json`

This will display a graph with matplotlib. Note that in order to stop the script, you need to press the red X on the figure, rather than stopping it from the console.

In order to display the data in `pgfplots` in LaTeX, we'll need to create an output directory to store a table file for each model. Run

`mkdir attack_data_final_real/pdfrate_jsma; mkdir attack_data_final_real/pdfrate_jsma/models_final_real`

then run

`python plot_float_robustness attack_data_final_real/pdfrate_jsma.json -o attack_data_final_real/pdfrate_jsma/models_final_real`

This will generate text files that are `pgfplots` tables. You can then import them as tables into `pgfplots` to plot them in LaTeX.

# Running the weight similarity experiment

To run the weight similarity experiment on hidost, first we make a directory to store the data

`mkdir similarity_out_final`

then, do

`python weight_similarity.py hidost -a custom_sigmoid -o similarity_out_final/hidost_similarity.json -n 64 -N 100000 -N 200000 -N 400000 -N 800000`

This says to run weight similarity on models trained on `hidost` using the `scaled_sigmoid` activation. It saves the results in `json` format to `similarity_out_final/hidost_similarity.json`. It says to train 64 models and use all of them for the experiment. It also says to use summation sizes of 100000, 200000, 400000, and 800000 to approximate the expectation.

# Viewing the output of the weight similarity experiment

To graphically display the output, run

`python plot_similarity.py --multi-in similarity_out_final/hidost_similarity.json -b 80`

This says take the data from `similarity_out_file/hidost_similarity.json`, which contains the results of approximations using 4 different summation sizes,
and graph them using 80 bins total.

Then, to save the files into tables for use in LaTeX `pgfplots`, do

`mkdir similarity_out/hidost_similarity`

`python plot_similarity --multi-in similarity_out_final/hidost_similarity.json -o similarity_out_final/hidost_similarity`












