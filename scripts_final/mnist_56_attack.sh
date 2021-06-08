python attack.py custom_jsma -d TEST_mnist_thresh,5,6 -i models_final_real/mnist_56_linear_sigmoid.h5:linear -m linear \
  -d TEST_mnist_thresh,5,6 -i models_final_real/mnist_56_linear_sigmoid_stable_bias.h5:linear -m linear \
  --all -o attack_data_final_real/mnist_56_jsma.json

