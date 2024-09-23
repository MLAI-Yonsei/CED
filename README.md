# CED
This is the official code of our paper CED: Comparing Embedding Differences for Detecting Out-of-Distribution and Hallucinated Text (EMNLP findings 2024).


# Requirements
In order to reproduce our results, first install the required dependencies:

    conda create -n CED python=3.9
    conda activate CED
    pip install torch==2.0.1
    pip install -r ./requirements.txt

This will create conda environment ```CED``` with correct dependencies.
## Data
For datasets used in classification task, download ```dataset.zip``` file from the [link](https://drive.google.com/file/d/1whsGbpWq5zkjHc80U28pPpnu2E028UpP/view?usp=drive_link) , and unzip the file under root directory. 

# Scripts
We provide scripts to run CED for clinc dataset. All datasets can be processed the same way by changing the ```output_dir```, ```dataset``` of the scripts to the matching dataset.

Run CED with a pre-trained model:

    bash scripts/clinc_pre.sh
    
Train the model on clinc dataset:

    bash scripts/clinc_train.sh
    
Run CED with a fine-tuned model:

    bash scripts/clinc_ft.sh

# Citation
If our repository is used in your research, we would greatly appreciate your acknowledgment through citation:

# Acknowledgements
Our repository relies on resources from [GNOME](https://github.com/lancopku/Avg-Avg), [FLatS](https://github.com/linhaowei1/FLatS) repository. We thank the authors (Sishuo Chen et al., Haowei Lin et al.) for sharing codes for extensive research.
