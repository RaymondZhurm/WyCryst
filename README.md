# WyCryst: Wyckoff Inorganic Crystal Generator Framework

WyCryst is a generative design framework for inorganic material crystal materials 


## Installation

Please use python 3.8.10 to run the model.

To install, just clone the repository. Then install all required packages:

```bash
pip install -r requirement.txt
```

Before running the model, please download the data file from GitHub.
Please add the df_allternary_newdata.pkl to temp_files\wyckoff_data folder

## Usage

You can train and test the PVAE model by:

```bash
python train.py
```

To reproduce our results in the Leave-one-out Validation Section, you can use pre-trained model by:

```bash
python validate.py
```

