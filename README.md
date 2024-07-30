# Der, Die, Das

In this repository, I implemented a simple model to predict the gender of a work by just looking at the word. It contains datasets for German, Croatian, and Catalan, but the model can be easily extended to other languages.

The idea is to use the model to answer the following research question: "How intuitive is the gendering of words in different languages?". If we train on different languages with the same settings (same dataset size, same architecture), we can compare the performance of the model on different languages, and thus get an idea of how intuitive, regular, or learnable are the nouns of that language. 

## Table of Contents

- [Installing and running](#installing-and-running)
- [License](#license)

## Installing and running

This project uses Hatch as the project manager. Therefore, you need to install it first:

```bash
pip install hatch
```

Hatch will create a virtual environment for you and install all the dependencies. You do not need to create a virtual environment manually.

To run the project, you can use the command
    
```bash
hatch run der_die_das [OPTIONS] COMMAND [ARGS]
```

To see the available commands, run ``hatch der_die_das --help``. To see the available options for a command, run ``hatch der_die_das COMMAND --help``.

The three commands are the following:
- ``process_data``: Run the data processing pipeline.
- ``train``: Train the model on the train dataset.
- ``evaluate``: Evaluate the model on a the test dataset.

For example, run: ``hatch run der_die_das train german`` to train the model on the German dataset with the default settings.


## Data
A nice byproduct of this project is that it contains "noun" datasets for German, Croatian, and Catalan. These are tables with > 2k nouns and their gender. This can be used to power flashcard-based language learning apps, or to train other models.

- German: 2539 unique nouns. Find it in ``data/german_words_processed.csv``.
- Croatian: 2196 unique nouns. Find it in ``data/croatian_words_processed.csv``.
- Catalan: 2250 unique nouns. Find it in ``data/catalan_words_processed.csv``.

The German dataset was obtained from this source:
- https://frequencylists.blogspot.com/2015/12/the-2000-most-frequent-german-nouns.html

The Croatian and Catalan datasets were obtained by translating the German dataset using the Microsoft Translator API (https://www.microsoft.com/en-us/translator).

_Note_: the datasets were not manually curated, so there might be some errors. For training a model, the quality is good enough, but for linguistic purposes, you might want to clean the datasets.

## License

`der-die-das` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
