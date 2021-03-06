from argparse import ArgumentParser

parser = ArgumentParser()

# parser.add_argument("dataset", action="store")
# parser.add_argument("in_file", action="store")
# parser.add_argument("-o", dest="out_file", action="store")
# parser.add_argument("-a", dest="accuracies", action="append")
# # parser.add_argument("--no-acc", dest="no_acc", action="store_true", default=False)
#
# args = parser.parse_args()
#
# dataset = args.dataset
# in_file = args.in_file
# out_file = args.out_file
# accuracies = args.accuracies
#

datasets = [
  "pdfrate", "hidost_scaled", "hatespeech"
]

in_files = ["models_relu/pdfrate.h5:relu", "models_relu/hidost.h5:relu", "models_relu/hatespeech.h5:relu"]


out_files = ["models_relu/pdfrate_{e}.h5", "models_relu/hidost_{e}.h5", "models_relu/hatespeech_{e}.h5"]

accuracies_list = [
  ["0.990", "0.985", "0.980", "0.975"],
  ["0.995", "0.990", "0.980", "0.975"],
  ["0.90", "0.89", "0.88", "0.87"]
]


from src.data import get_data


for dataset, in_file, out_file, accuracies in zip(datasets, in_files, out_files, accuracies_list):

  assert("{e}" in out_file)

  x_train, y_train, x_test, y_test = get_data(dataset)

  # all these imports take a while so we do them after we see that the data returns correctly

  import tensorflow as tf
  from tensorflow import keras

  from src.models import load_model, load_mnist_model, sign, load_general_model
  from src.stabilization import stabilize_some_l1

  # accuracies are sorted so that we can just "keep going" after hitting 1 threshold

  acc_string_and_values = [(float(a), a) for a in accuracies]
  acc_string_and_values.sort(key=lambda x: x[0], reverse=True)

  model, _ = load_general_model(x_train, y_train, in_file, 1024, None, None)

  print(acc_string_and_values)

  already_changed = set()

  delta_robs = {}

  delta_rob_sum = 0
  for acc, acc_string in acc_string_and_values:

    # stabilize it
    model, already_changed, delta_rob = stabilize_some_l1(model, x_test, y_test,
                                               thresh=acc, allowed_layers=(0, 2), no_accuracy=True, already_changed=already_changed)



    delta_rob_sum += delta_rob

    print(f"delta_rob {delta_rob_sum}")

    delta_robs[acc_string] = delta_rob_sum

    # save it
    out_file_name = out_file.replace("{e}", acc_string)

    print(f"Accuracy before saving:")
    model.evaluate(x_test, y_test)
    model.save_weights(out_file_name)
    print(f"saved to {out_file_name}")





