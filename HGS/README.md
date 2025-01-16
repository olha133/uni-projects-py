# General-to-Specific Heuristic for Classification

## Overview
The General-to-Specific Heuristic (GSH) creates a trade-off between optimality and unrestricted search. While the concept descriptions produced by GSH are not guaranteed to cover all positive examples and exclude all negative examples, this approach prioritizes heuristic search effectiveness.

## Datasets Used
Two datasets were used for this task:

1. **Dataset from exercises** (non-numeric): This dataset is relatively small and simple, and it served as the basis for the program.
   
2. **COVID-19 Symptom Dataset** (numeric): This dataset contains information about 10 COVID-19 symptoms and their severity classification. 

### COVID-19 Symptom Dataset Details
- **Features**: The dataset includes 10 symptoms:
  - Fever
  - Tiredness
  - Dry-Cough
  - Difficulty-in-Breathing
  - Sore-Throat
  - None Symptom
  - Pains
  - Nasal-Congestion
  - Runny-Nose
  - Diarrhea
- **Classes**: The dataset is divided into two severity classes:
  - **Severity None** (NSET): 12 cases
  - **Severity Severe** (PSET): 15 cases
- **Total Records**: 27

This dataset was used to demonstrate the application of the General-to-Specific Heuristic for classification and decision-making tasks.
## Functions

### 1. `prepare_cv_data()`
This function prepares the dataset used for the exercise. It processes the data by:
- Splitting the dataset into positive PSET examples and negative NSET examples.
- Creating the most general HSET example.

### 2. `prepare_covid_data()`
This function prepares a dataset containing COVID-19 symptoms. It:
- Splits the dataset into positive PSET (Severe) cases and negative NSET (None) cases.
- Creates the most general HSET example for the given data.
- Performs several data visualizations to assist with understanding and analyzing the dataset.

- ### 3. `specify(H)`
**Input**: One example from HSE  
**Output**: All possible specifications of this SPECS example  
This function specifies an example of H by removing one element.

### 4. `score(S, PSET, NSET)`
This function calculates the score using a predefined formula, evaluating the performance of a given specification in relation to positive and negative examples.

### 5. `is_as_specific(S, C)`
This function checks if two specifications, `S` and `C`, are at the same specification level.

### 6. `hgs(PSET, NSET, CLOSED_SET, HSET, beam_size)`
This is the most important function of the General-to-Specific Heuristic (HGS). It combines all the previous functions and finds a specification for positive examples from a given set of data, implementing the heuristic search process.

### 7. `evaluate(result, PSET, NSET)`
This function calculates various performance metrics, including:
- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)
- Precision
- Recall
- F1 Score
- Correctness
- True Positive Rate
- True Negative Rate
- Positive Predictive Value (PPV)
- Negative Predictive Value (NPV)
- Error Estimate  

Additionally, it creates and visualizes a confusion matrix to aid in evaluating the model's performance.
