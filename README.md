# Welcome!
Here, we'll explain how to get the testing environment setup and running the experiments that we used to generate the graphs in our paper.

These experiments were run on a research server running linux using Intel Xeon Gold 6150 CPU's running at 2.70GHz. This codebase is not guaranteed to run on
any other machine, and in fact tensorflow is known to fail on some architectures (if the python script terminates with "Illegal Exception", this
is most likely why").

# Environment Setup
We used Python 3.7.7 and pip 19.2.3. Any version of python above 3.6 _should_ work, but has not been tested.
