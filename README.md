# Evolving Stream Classification Challenge

The field of stream data mining has become increasingly important due to the high speed of data generation and the need for fast and accurate information processing and decision-making. This project focuses on implementing and comparing common stream data mining methods, including four ensemble models and a neural network, trained on multiple data streams.

The ensemble methods used in this project include **Adaptive Random Forest**, **Streaming Agnostic Model with k-Nearest Neighbors**, **Streaming Random Patches**, and **Dynamic Weighted Majority**. The performance of these ensemble methods is compared to the performance of a **Multi Layer Perceptron** neural network model using a test-then-train prequential evaluation method.

To gain a comprehensive understanding of the performance of these methods, both synthetic and real datasets are used. Real data streams are obtained from the **Spam** and **Rialto** datasets, which are publicly available ([access](https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow)). Synthetic datasets are generated using the **HyperplaneGenerator** and **SEAGenerator** classes from scikit-multiflow to create 10,000 samples of each dataset. The results and generated data of these experiments are saved in a CSV file for further testing and analysis.

## Getting Started

1. Download or Clone the repository:</br>  ```git clone https://github.com/MPCL5/stream-data-challenge.git```
2. Navigate to project root. example: </br> ```cd stream-data-challenge```
3. Install the required dependencies:</br> ```pip install -r requirements.txt```
4. Run the project: </br> ```python evaluator.py```

### Prerequisites

- numpy
- scikit-learn
- scikit-multiflow

### Installing

Installation using pip suggested: </br>
```pip install -r requirements.txt```

### Running the code

After installing dependencies run `evaluator.py`. Prior to evaluation, the script 
automatically downloads real datasets if they are not downloaded then 
generates synthetic if they are not available locally. while evaluation is in progress
the results are being saved in `./Results` directory.