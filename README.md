# Ad detection test task

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the packages.

```bash
pip install -r requirements
```

## Usage

### Training the model

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip -d glove
python train_model.py --path ml_interview_ads_data/
```

### Running inference on a test set

```bash
python run_model.py --path ml_interview_ads_data/
```
