# Master Thesis Repository

This repository contains the code for the Master's thesis: **"Encoder-Decoder Transformer for Speech Neuroprosthesis Trained on Multiple sEEG Datasets."** The thesis can be found in the main directory, with the LaTeX files stored inside the `latex` directory.


## Directory Structure
- **`/preprocessing`**: Contains scripts for preprocessing neural recordings, extracting sEEG and audio features.
- **`/models`**: Includes transformer and autoencoder models along with scripts containg various helper classes and functions.
- **`/training_files`**: Contains the training files used for model training.
- **`/results`**: Stores results from experiments.
- **`/StreamingVocGan`**: Contains the pretrained VocGan model for audio processing.
- **`/sentences_data`**: should contain the `.xdf` files.

## Preprocessing
In the `/preprocessing` directory, various scripts are available to process the neural recordings. To load and preprocess the data, use the `loading_files.load_data()` function. Ensure that the `.xdf` files are located inside `./sentences_data/sentences`. Additionally, this function requires a configuration dictionary. For an example, look inside any of the run scripts found in the main directory.

## Model Training
The transformer and autoencoder models, along with their respective helper functions, can be found in the `/models` directory. To run the training for these models, execute the following scripts:
- `run_autoencoder.py`
- `run_transformer.py`
- `run_pretrained_transformer.py`

These scripts will call the training files located in the `/training_files` directory. 

### Configuration
To configure the architecture of the transformer model, modify the configuration dictionary within the `run_...` scripts. To obtain latent space visualizations, ensure to select this option in the configuration dictionaries.

## StreamingVocGan
The `/StreamingVocGan` directory contains a pretrained VocGan model, which can extract mel-spectrograms from raw audio and synthesize predicted mel-spectrograms back into audio.

## Results
Results for most experiments can be found in the `/results` directory.

## Important Note
By default, the training scripts save extensive training and evaluation data in the main directory. It is advisable to call the main function in the training scripts with an output path specified for all runs.

## Contributions
The following scripts were provided by Joaquín Amigó Vega from the Neural Interfacing Lab:
- `preprocessing_audio.py`: Contains code for extracting mel-spectrograms and synthesizing speech.
- `preprocessing_generic.py`: [Add a brief description here]
- `preprocessing_markers.py`: [Add a brief description here]
- `preprocessing_neural.py`: Contains code for calculating neural features.
- `preprocessing_trajectories.py`: [Add a brief description here]
- `loading_files.py`: Main script for the preprocessing pipeline; loads and saves the raw data and features.
