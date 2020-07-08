# PSAM
This is the python implementation of PSAM model for paper "Personalized, Sequential, Attentive, Metric-Aware Product Search"

## Datasets
Download Amazon review datasets from [http://jmcauley.ucsd.edu/data/amazon/](http://jmcauley.ucsd.edu/data/amazon/)(In our paper, we used 5-core review data and metedata).

## Run
run the main.py with the command:<br/>
python main.py --model './Model/psammodel/' --ReviewDataPath './Dataset/Review/' --MetaDataPath './Dataset/Meta/' --window-size 5
