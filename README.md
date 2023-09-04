# NN-CA1-McCullochPitts-Adaline-Madaline-AutoEncoders-MLPS

Neural Networks & Deep Learning Course, University of Tehran

## Part 1: McCulloch-Pitts Neural Network

In this section, we'll explore the state transition table associated with the McCulloch-Pitts neuron network. It's important to note that in all neurons, there will be three columns on the left side of the table, representing inputs. Each of the right-side columns corresponds to the output of a neuron.

To simplify the process, we'll start by plotting the truth table for each neuron's output.

We define the network as a Python class that takes weights as a two-dimensional matrix. In the future, by providing an array of inputs, it returns the network's final output using suitable matrix multiplication and comparing the outputs with a threshold.

## Part 2: Adaline and Madaline Networks

### Adaline (1-2)

- Create two matrices: the first one is 1x100 with a mean of 0 and a standard deviation of 0.1. The second one is 1x100 with a mean of 0 and a standard deviation of 0.4.

- Create another two matrices: the first one is 2x100 with a mean of 1 and a standard deviation of 0.2. The second one is 2x100 with a mean of 0 and a standard deviation of 0.2.

In Figure 9, you can see that the defined datasets can be separated with a single adaline line and even, based on this figure, it's possible to bring the loss close to zero (depending on how you define the loss).

### Adaline (2-2)

- In this section, we define an Adaline model with two inputs and one output, similar to the previous part. Since the data is not correlated, and there is no distribution, and the data from each group is not clustered together in different parts of space, we can effectively separate the data using Adaline.

As seen in the figure, the second dataset has more spread and is closer together, making it impossible to separate all the data from each other with a single adaline. Therefore, we need models that perform better than Adaline, meaning we need to find Madaline models.

### Madaline

- In MRI, weights are adjusted for hidden adalines, and you need to learn that. Weights for the output unit are fixed and do not need to be learned in the training process; we fix them according to the problem's needs. While in MRII, a method for setting and learning all weights in the network is considered.

As shown in Figure 15, it is evident that a single Adaline cannot separate data from two classes, and we need multiple Adalines, hence the need for Madaline models.

As seen in Figure 16, Madaline with three neurons cannot reduce the error significantly beyond a point because it lacks the necessary power to separate the data from two classes. However, Madaline with four neurons can perform much better, and according to Figure 17, it can even bring the loss to zero. Madaline with ten neurons can also achieve zero loss, as shown in Figure 18. However, it suffers from redundancy, as evident in Figure 21.

Figure 22 illustrates the number of epochs for three models (3, 4, and 10 neurons).

For the number of epochs, the model with three neurons could not satisfy the condition of weight non-change, and it finished with the second condition, which was a maximum of 300 epochs. But the model with four neurons reached the condition of weight non-change with 16 epochs, and the model with ten neurons reached it with 44 epochs, as shown in the figure.

## Part 3: Implementing LMAE for Classification (Based on LMAE Article)

### Introduction and Data Preprocessing

In this section, we will implement the code based on the research article titled "LMAE: A large margin Auto-Encoders for classification" available at [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0165168417302013).

To begin with, we load the data using the Keras library and plot the desired graphs based on the labels of the training data.

Next, we randomly visualize five data points. To normalize the inputs, we use max-min normalization, ensuring that all data points fall between 0 and 1. Since the input data consists of black and white images with pixel values ranging from 0 to 255, we can directly divide the data by 255.

### Auto-Encoder Network

We implement the mentioned network, consisting of Encoder and Decoder, as Sequential models. The Encoder-Auto network is created by connecting these two models. We experiment with various activation functions for the layers and find that the best performance is achieved when using sigmoid for the output layer and Relu-Leaky for the other layers. Since we expect the output to be in the range of 0 to 1, choosing these activation functions makes sense. We train the model for 10 epochs. The results show that the loss curve decreases nicely, and the training process can be stopped. To ensure the correct functioning of the Encoder-Auto network, we visualize the input and output for a few random data points from the test data. We expect the output, reconstructed by the Decoder using the 30 features from the Encoder, to closely resemble the original image.

### Classification

We utilize the Encoder from the previous section and project all the training and test data into the 30-dimensional feature space. Then, we implement the classification network as required, with two hidden layers containing 24 and 16 neurons, and an output layer with 10 neurons (since we have 10 classes). The network is trained using the data in the new feature space for 10 epochs. The requested plots show the loss, validation loss, and accuracy during the training process. The classification accuracy after completing the training process is approximately 94.8%.

Finally, we write a function for classification. Given input in the original space (784 dimensions), it first maps it to the 30-dimensional feature space using the Encoder and then determines its label using the trained classifier. We visualize the outputs of the Encoder and the classifier together for 5 random data points from the test data. We also present a confusion matrix plot to illustrate the classification results.

Due to the good accuracy of the classifier, most data points are located on the main diagonal of the confusion matrix, indicating correct classification. However, upon closer examination of misclassifications, we find interesting results. For example, digits 4 and 9, which share some similarities, are frequently confused with each other.

## Part 4: Multi-Layer Perceptron

In this section, we will cover the data preprocessing steps, including data exploration, cleaning, feature engineering, and model evaluation. We will also delve into the results obtained from various models.

### 4-1 Data Exploration and Preprocessing

- **Reading CSV and Data Information**: We began by reading the CSV file and inspecting the dataset. It comprises 25 features related to automobiles, with the target variable being the price. Our primary objective is to predict car prices based on these features.

- **Handling Missing Data**: We assessed the dataset for missing values using the `isna` function. Fortunately, no missing data was found in any of the columns.

- **Removing Unnecessary Columns**: Three columns, namely "CarName," "ID_car," and "symboling," were identified as unnecessary for our analysis and were removed.

### 4-2 Data Analysis and Visualization

- **Exploring Correlation**: To understand the relationships between features and the target variable, we computed the correlation matrix. The feature with the highest correlation to car prices was "enginesize," indicating its significance for price prediction.

- **Visualizing Data**:
  - **Price vs. Engine Size**: A scatter plot revealed a linear relationship between the "enginesize" feature and car prices.
  - **Price Distribution**: We examined the distribution of car prices, which showed that the dataset is not biased towards luxury cars, as prices span a wide range.

### 4-3 Data Transformation

- **Converting Categorical Data to Numeric**: To prepare the data for modeling, we transformed categorical features into numeric ones using one-hot encoding. This process resulted in the creation of 37 additional columns.

### 4-4 Data Splitting

- **Splitting Data**: We divided the dataset into three sets: training (70% of the data), validation (15%), and test (15%) sets. This splitting allows us to train, tune, and evaluate our models effectively.

### 4-5 Data Scaling

- **Scaling Data**: We applied Min-Max scaling to normalize the feature values, ensuring that all features have similar scales.

### 4-6 Model Selection and Evaluation

- **Selecting Model Architecture**: We experimented with various Multi-Layer Perceptron (MLP) models, varying the number of hidden layers. The models utilized ReLU activation functions for hidden layers and linear activation for the output layer. Dropout layers were added after each hidden layer to mitigate overfitting.

- **Choosing Loss Functions and Optimizers**: We explored different loss functions and optimizers to identify the best combination for our regression problem. The selected loss function was Mean Absolute Percentage Error (MAPE), which suits regression tasks. We tested both Stochastic Gradient Descent (SGD) and Adam optimizers.

### Results

- The "enginesize" feature demonstrated the highest correlation with car prices, making it a key predictor.
- Price distribution revealed no bias towards luxury cars.
- Data preprocessing, including one-hot encoding and scaling, prepared the dataset for modeling.
- Data was effectively split into training, validation, and test sets.
- Model evaluation revealed that a three-layer MLP with MAPE loss and SGD optimizer achieved the best performance, with a 74% R-squared score.
