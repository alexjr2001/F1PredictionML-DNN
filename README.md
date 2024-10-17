# Formula 1 Position Prediction: Comparison of Methods with and without Batch Normalization

The primary objective of this project is to **predict the final positions** of drivers in Formula 1 races using two different approaches based on deep neural networks. We compare two models: one that utilizes **Batch Normalization** and another that does not, in order to evaluate which method provides better results in terms of prediction accuracy and performance.

## Project Structure

The project is divided into the following phases:

### 1. Data Preprocessing (`datasetPreProcessing.ipynb`)

- **Objective**: Clean and transform raw data into a format ready for training the models.
- **Data Used**: Historical Formula 1 race data, including results, driver information, and constructor data.
- **Key Steps**:
  - Import relevant datasets.
  - Select key features such as `resultId`, `raceId`, `driverId`, `constructorId`, `grid`, and `positionOrder` (final position).
  - Merge data from various files to add additional information like driver names and constructor details.
  - Convert categorical data (e.g., driver and constructor names) into numerical values for model input.
  - **Target Variable**: The goal is to predict the `positionOrder` column, which represents the final position of the driver in each race.

### 2. Model Training (`trainingModel.ipynb`)

- **Objective**: Train and compare two deep neural network models to predict the final position of a driver in a Formula 1 race.
- **Models**:
  - **Model 1**: Deep neural network **with Batch Normalization**, which includes normalization layers after each hidden layer.
  - **Model 2**: Deep neural network **without Batch Normalization**.
- **Approach**:
  - Define and compile both models using **TensorFlow/Keras**.
  - Train both models on the same preprocessed dataset.
  - Evaluate the performance of both models using metrics like **accuracy in predicting final positions**, convergence speed, and training time.

## Dataset

The dataset used comes from several Formula 1-related sources, including:

- `results.csv`: Contains race results and key metrics such as final positions.
- `drivers.csv`: Provides information about drivers.
- `constructors.csv`: Contains constructor information.

The preprocessing step merges these data sources to create a comprehensive dataset with relevant features for training the models. The **target variable** is `positionOrder`, which represents the driver's final race position.

## Dependencies

- Python 3.x
- Pandas
- NumPy
- TensorFlow / Keras
- Matplotlib (for plotting performance)

You can install the required packages with the following command:

```
pip install pandas numpy tensorflow matplotlib
```

## How to Run the Project

1. **Preprocessing the Data**:
   - Open the `datasetPreProcessing.ipynb` notebook.
   - Run all cells to preprocess the data and save it for training.

2. **Training the Models**:
   - Open the `trainingModel.ipynb` notebook.
   - Run the notebook to train both deep neural network models.
   - Compare the results to understand the effect of Batch Normalization on predicting the final positions of Formula 1 drivers.

## Results

- This project evaluates the impact of Batch Normalization on training speed, convergence, and accuracy in predicting race positions.
- Results are visualized through loss and accuracy plots for both models.

## Future Work

- Experiment with different hyperparameters such as learning rate, batch size, and the number of epochs.
- Test the models on other datasets to generalize the findings.
- Explore the impact of additional techniques like dropout or data augmentation.

## License

This project is licensed under the MIT License.
