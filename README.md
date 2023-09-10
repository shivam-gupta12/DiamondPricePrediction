# Diamond Price Prediction

This repository contains a machine learning model for predicting the price of diamonds based on a set of features. The dataset used for this project consists of approximately 54,000 diamonds, each described by 10 attributes, with the target variable being the price in US dollars.

## Dataset Description

### Features

1. **Carat**: The carat is the diamond's physical weight, measured in metric carats. One carat equals 1/5 gram and is subdivided into 100 points. Carat weight is the most objective grade of the 4Cs.

2. **Cut**: The quality of the cut is determined by evaluating the cutter's skill in the fashioning of the diamond. The more precise the diamond is cut, the more captivating it is to the eye. Cut quality can be one of the most critical factors influencing a diamond's beauty.

3. **Color**: Diamond color is graded on a scale from J (worst) to D (best). The color of gem-quality diamonds occurs in various hues, ranging from colorless to light yellow or light brown. Colorless diamonds are the rarest and most valuable.

4. **Clarity**: Clarity refers to the presence of internal characteristics known as inclusions or external characteristics known as blemishes. The clarity scale ranges from I1 (worst) to IF (best). Diamonds without inclusions or blemishes are exceptionally rare and valuable.

5. **X**: Length in mm (0.0 to 10.74) - Represents the length of the diamond.

6. **Y**: Width in mm (0.0 to 58.9) - Represents the width of the diamond.

7. **Z**: Depth in mm (0.0 to 31.8) - Represents the depth of the diamond.

### Target

- **Price**: The target variable is the price of the diamond in US dollars, ranging from $326 to $18,823.

## Project Overview

The goal of this project is to build a machine learning model that can predict the price of diamonds based on these diamond characteristics. To achieve this, we have used various data preprocessing techniques and machine learning algorithms to train and evaluate the model's performance.

## Usage

1. Clone this repository:

   ```
   https://github.com/shivam-gupta12/DiamondPricePrediction.git
   ```

2. Navigate to the project directory:

   ```
   cd DiamondPricePrediction
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook or Python script to train and evaluate the machine learning model.

## Data Source

The dataset used in this project is a classic dataset containing diamond prices and attributes, originally sourced from [https://www.kaggle.com/code/karnikakapoor/diamond-price-prediction]. You can find the dataset in the `data` directory.

## Contributors

- Shivam Gupta (https://github.com/shivam-gupta12) - Project Lead

## Acknowledgments

We would like to express our gratitude to the creators of the dataset and the open-source community for providing valuable resources for data science and machine learning projects.

Feel free to contribute to this project by improving the model, adding new features, or enhancing the documentation. Happy coding!
