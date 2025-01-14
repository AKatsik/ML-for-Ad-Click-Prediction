# ML for Ad Click Prediction

ML pipeline of a binary classification model to predict whether an ad will be clicked (click/no click).

This is an 8-hour project for showcasing ML and OOP skills only.\
Some steps for further development are outlined below, as they could not be included due to time constraints.
- Collaborate with domain experts to get better data understanding and optimise decision making
- Integrate/Use feature importance to optimise the existing model or develop a new one
- Further hyper-parameter tuning to improve model performance and reduce underfitting
- Integrate under-sampling and/or over-sampling techniques to address class imbalance
- Develop more advanced classification algorithms such as Gradient Boosting Trees and XGBoost 
- Avoid existing mistake to maximise minority class availability

## Datasets

1. **Main-dataset**:  
   Used for training, validation, and initial testing.  
   - Highly imbalanced, leading to specific design decisions.
   - Split into training and test sets, ensuring similar proportions of the minority and majority classes in both sets.
   - The test set size represents 50% of the Main-dataset to maintain an adequate number of minority class samples.

2. **Unseen-dataset**:  
   Used only for final model evaluation.  
   - Actual targets for the unseen data are not provided.

---

## Key Considerations for Imbalanced Data

1. **Data Splitting**:  
   The Main-dataset is split to preserve the class proportions in both training and test sets.

2. **Class Weights**:  
   Weights are calculated based on class frequencies and applied during training to penalize the majority class.  
   - Alternative techniques like undersampling or oversampling could be explored if more development time were available.

3. **Evaluation Metric**:  
   Precision-Recall Curve is used as the primary evaluation metric.  
   - Recommended for imbalanced datasets due to its focus on the performance of the minority class.

---

## Steps to Run the Pipeline

### Step 1: Navigate to the project directory
Use the following command to navigate to the directory where you would like to store and run the project:
- **On Windows/macOS/Linux**:
   ```bash
   cd [path]
   ```

### Step 2: Prepapre a Python virtual environment
Use the following command to create a virtual environment in the project root directory:
- **On Windows**:
   ```bash
   python -m venv [venv name]
   ```

- **On macOS/Linux**:
   ```bash
   python3 -m venv [venv name]
   ```

### Step 3: Activate the virtual environment
Use the following command to activate the virtual environment (assuming that the venv name is "myenv"):
- **On Windows**:
   ```bash
   myenv\Scripts\activate
   ```

- **On macOS/Linux**:
   ```bash
   source myenv/bin/activate
   ```

### Step 4: Install all the necessary Python packages  
Use the following command to install all the necessary Python packages in the virtual environment:  
- **On Windows/macOS/Linux**:  
  ```bash
  pip install -r requirements.txt
  ```

### Step 5: Run the Pipeline  
Execute the main.py file to run the pipeline:  
- **On Windows**:  
  ```bash
  python main.py
  ```

- **On macOS/Linux**:  
  ```bash
  python3 main.py
  ```