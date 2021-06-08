num_rounds = 20
num_epochs = 2
dataset = "pdfrate"
out_path = "stabilize_some_results.json"

sizes = [16, 64, 256, 1024, 4096, 16384, 65536]
betas = [0.99, 0.98, 0.97]

from src.data import get_data

from src.models import load_general_model



x_train, y_train, x_test, y_test = get_data(dataset)

results = [[0 for _ in betas] for _ in sizes]

for size_index, size in enumerate(sizes):
    # train a model with the given layer size
    import tempfile
    import os

    model, _ = load_general_model(x_train, y_train, None, size, None, None, "custom_sigmoid")

    model.summary()

    with tempfile.TemporaryDirectory() as dir:

        best_validation_acc = 0
        best_model_path = os.path.join(dir, "best_model.h5")

        for i in range(num_rounds):

            model.fit(x_train, y_train, epochs=num_epochs, verbose=0)
            _, validation_acc = model.evaluate(x_test, y_test, verbose=0)

            if validation_acc > best_validation_acc:
                best_validation_acc = validation_acc
                model.save_weights(best_model_path)
                print(f"saved with acc {validation_acc}")

        model.load_weights(best_model_path)
        model.evaluate(x_test, y_test)

        # now, reload the model every time and do the stabilization tests
        for beta_index, beta in enumerate(betas):
            model.load_weights(best_model_path)


            from src.stabilization import stabilize_logn
            from time import process_time
            # start timer
            start = process_time()
            model, changed, _ = stabilize_logn(model, x_test, y_test, thresh=beta, allowed_layers=(0, 2), verbo=2)
            # end timer
            end = process_time()
            model.evaluate(x_test, y_test, verbose=2)

            print(f"num change: {len(changed)}")
            print(f"model size: {model.get_weights()[0].size}")

            duration = end - start

            results[size_index][beta_index] = duration



        with open(out_path, "w") as out_file:
            import json

            json.dump({"results": results, "sizes": sizes, "betas": betas, "dateset": dataset}, out_file)







