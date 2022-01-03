# Graph Transformer
Graph Transformer

To run the model make sure you have a figs folder in which all the figures will be stored.

To install dependencies execute:
pip install -r requirements.txt

To train the model:
python3 train.py

The flags of train.py control the dataset, model and parameters.

The structure of the code is as follows:

    src/
      |- models.py File which contains all of the models
      |- modules.py File which contains modules for the Transformer code
    train.py Main script that train and evaluates the models
    utils.py utility functions for the train.py file