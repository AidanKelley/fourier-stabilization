import csv
import random

filename = "hatespeech/gab.csv"
filename_train = "hatespeech/gab_train.libsvm"
filename_test = "hatespeech/gab_test.libsvm"

n_top = 1000

all_the_bags = []
hate_y = []
test_train_assignments = []

with open(filename, "r") as file_in:
    # headers: id, text, indices that are hatespeech (1 indexed), responses
    reader = csv.reader(file_in)

    rows = (row for index, row in enumerate(reader) if index != 0)
    count = 0

    word_freqs = {}

    for row in rows:

        all_text = row[1]
        bags_of_words = [
            [
                word.lower()
                for index, word in enumerate(line.replace("\t", "").split(" "))
                if index != 0
            ]
            for line in all_text.split("\n")
        ]

        number_of_entries = len(bags_of_words)

        hate_indices = row[2]

        is_it_hate = [1 if str(index + 1) in hate_indices else 0
                      for index in range(number_of_entries)]

        the_split = [random.randint(0, 1) for _ in bags_of_words]

        all_the_bags += bags_of_words
        hate_y += is_it_hate
        test_train_assignments += the_split

        # only use words in the train partition
        all_words = (word for index, bag in enumerate(bags_of_words)
                     if the_split[index] == 0
                     for word in bag)

        for word in all_words:
            if word in word_freqs:
                word_freqs[word] += 1
            else:
                word_freqs[word] = 1


number_of_bags = len(all_the_bags)

word_freq_pairs = [(word, freq) for word, freq in word_freqs.items()]

# get the top n words from the training set
word_freq_pairs.sort(key=lambda x: x[1], reverse=True)
top_n = word_freq_pairs[0:n_top]

used_words_and_indices = {word: index for index, (word, _) in enumerate(top_n)}

# now, we'll generate the output data

with open(filename_train, "w") as train_file:
    with open(filename_test, "w") as test_file:
        for bag_index, bag in enumerate(all_the_bags):
            word_included = [False for _ in range(n_top)]

            for word in bag:
                if word in used_words_and_indices:
                    word_index = used_words_and_indices[word]
                    word_included[word_index] = True

            line_array = [f"{index}:1" for index, included in enumerate(word_included)
                          if included]

            y = hate_y[bag_index]
            line = str(y) + " " + " ".join(line_array) + "\n"

            # split into two files
            if test_train_assignments[bag_index] == 0:
                train_file.write(line)
            else:
                test_file.write(line)