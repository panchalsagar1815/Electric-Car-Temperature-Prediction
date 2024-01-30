# Elect
**Project Title: Electric Car Temperature Prediction**

**Overview:**
The "Electric Car Temperature Prediction" project aims to develop a predictive model that forecasts temperature changes in electric cars based on various features such as ambient conditions, coolant levels, motor speed, torque, and more. By employing deep learning techniques and a convolutional neural network (CNN), the project enhances the accuracy of temperature predictions, contributing to the efficiency and performance of electric vehicles.

**Key Features:**

1. **Data Collection and Cleaning:**
   - Compiled a dataset containing key features such as ambient conditions, coolant levels, motor speed, torque, and others.
   - Employed data cleaning techniques to handle missing values, and outliers, and ensure data consistency.
   - Ensured the dataset's integrity for accurate temperature predictions.

2. **Data Preprocessing, Scaling, and Normalization:**
   - Preprocessed the data to handle sequential aspects and ensure compatibility with the neural network architecture.
   - Applied scaling techniques to normalize numerical features and improve model convergence.
   - Normalized the data to bring all features to a standard scale, preventing dominance by any particular feature.

3. **Data Visualization:**
   - Utilized data visualization techniques to gain insights into the distribution and relationships between different features.
   - Visualized key features to understand patterns and potential correlations that influence temperature changes.

4. **Neural Network Model:**
   - Implemented a convolutional neural network (CNN) for temperature prediction.
   - The model, described in the provided code snippet, consists of convolutional layers for feature extraction and linear layers for predicting temperature changes.
   - Utilized ReLU activation functions to introduce non-linearity and improve the model's capacity to learn complex patterns.

5. **Model Training and Evaluation:**
   - Trained the neural network on the preprocessed and normalized dataset.
   - Evaluated the model's performance using relevant metrics such as mean squared error (MSE) or mean absolute error (MAE) to quantify the accuracy of temperature predictions.

**Model Architecture:**
```python
class Model(nn.Module):
    def __init__(self, sequence_length, n_features):
        super(Model, self).__init__()
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        
        ''' Convolutional Layer'''
        self.features = nn.Sequential(nn.Conv1d(n_features, 16, kernel_size=3), nn.ReLU(), nn.Conv1d(16,32, kernel_size=1))
        
        self.lin_in_size = self.get_lin_in_size()
        ''' Linear Model'''
        self.predictor = nn.Sequential(nn.Linear(self.lin_in_size,30), nn.ReLU(), nn.Linear(30, 1))
        
    def forward(self, x):
        
        x = self.features(x)
        x = x.view(-1, self.lin_in_size)
        x = self.predictor(x)
        return x
    
    def get_size(self):
        rand_in = torch.rand(10, self.n_features, self.sequence_length)
        rand_out = self.features(rand_in)
        return rand_out.shape[-1] * rand_out.shape[-2]
```

**Conclusion:**
The "Electric Car Temperature Prediction" project introduces a sophisticated neural network model for forecasting temperature changes in electric cars. By incorporating features such as ambient conditions, coolant levels, and motor parameters, the model enhances the accuracy of temperature predictions. This project contributes to the advancement of electric vehicle technology, enabling more efficient temperature management and improved performance in varying conditions. Future work may involve fine-tuning the model, expanding the dataset, and incorporating real-time temperature sensor data for enhanced predictions in practical scenarios.
