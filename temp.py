# Script for generating data from different probability distributions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import os

# Combine features and labels into a single DataFrame
def save_dataframe(X : np.ndarray, y : np.ndarray, name: str):

    feature_cols = [f"feature_{i}" for i in range(1, X.shape[1] + 1)]
    beta_df = pd.DataFrame(X, columns=feature_cols)
    beta_df['target'] = y.astype(int)

    os.makedirs("./data", exist_ok=True)
    csv_path = f"./data/{name}.csv"
    beta_df.to_csv(csv_path, index=False)
    print(f"Saved dataframe to {csv_path}")


def generate_beta_dataset(n_samples = 6000, alpha = 2, beta = 5, n_features = 4, n_classes = 2):

    #Generate features from beta distribution
    X = np.random.beta(alpha, beta, size=(n_samples, n_features))

    #Create meaningful featrure interactions

    # Create meaningful feature interactions
    # Example scenario: medical diagnosis based on multiple measurements


    risk_score = (X[:, 0] * 2.0 +           # Primary risk factor
                  X[:, 1] * X[:, 2] * 3.0 +  # Interaction effect
                  np.where(X[:, 3] > 0.7, 1.5, 0) +  # Threshold effect
                  (X[:, 0] - X[:, 1])**2)    # Discrepancy effect
    
    threshold = np.percentile(risk_score, 60)  # Top 40% are class 1
    y = (risk_score > threshold).astype(int)

    """ non_linear_func = (X[:, 0]**2 + 
                       np.sin(2*np.pi*X[:, 1]) + 
                       X[:, 2] * X[:, 3] +
                       np.exp(X[:, 0] + X[:, 2]))
    
    threshold = np.median(non_linear_func)
    y = (non_linear_func > threshold).astype(int) """
    
    
    return X, y 



def generate_normal_dataset(n_samples = 6000, alpha = 2, beta = 5, n_features = 4, n_classes = 2):

    #Generate features from beta distribution
    X = np.random.standard_normal([n_samples, n_features])

    #Create meaningful featrure interactions

    # Create meaningful feature interactions
    # Example scenario: medical diagnosis based on multiple measurements


    risk_score = (X[:, 0] * 2.0 +           # Primary risk factor
                  X[:, 1] * X[:, 2] * 3.0 +  # Interaction effect
                  np.where(X[:, 3] > 0.7, 1.5, 0) +  # Threshold effect
                  (X[:, 0] - X[:, 1])**2)    # Discrepancy effect
    
    threshold = np.percentile(risk_score, 60)  # Top 40% are class 1
    y = (risk_score > threshold).astype(int)

    """ linear_func = (X[:, 0]*4 + 
                       X[:, 1]*2 + 
                       X[:, 2] * (-2) +
                       X[:, 3] * 3)
    
    threshold = np.median(linear_func) """
    # y = (linear_func > threshold).astype(int)
    
    
    return X, y 

def generate_poisson_dataset(n_samples = 6000, alpha = 2, beta = 5, n_features = 4, n_classes = 2):

    #Generate features from beta distribution
    X = np.random.poisson(15, size=[n_samples, n_features])

    #Create meaningful featrure interactions

    # Create meaningful feature interactions
    # Example scenario: medical diagnosis based on multiple measurements


    """ risk_score = (X[:, 0] * 2.0 +           # Primary risk factor
                  X[:, 1] * X[:, 2] * 3.0 +  # Interaction effect
                  np.where(X[:, 3] > 0.7, 1.5, 0) +  # Threshold effect
                  (X[:, 0] - X[:, 1])**2)    # Discrepancy effect
    
    threshold = np.percentile(risk_score, 60)  # Top 40% are class 1
    y = (risk_score > threshold).astype(int) """

    linear_func = (X[:, 0]*1 + 
                       X[:, 1]*1 + 
                       X[:, 2] * 1 +
                       X[:, 3] * 1)
    
    threshold = np.median(linear_func)
    y = (linear_func > threshold).astype(int)
    
    
    return X, y 




# def interactive_features_4d(n_samples=1000):
#     """
#     Binary classification with feature interactions (more realistic)
#     """
#     X = np.random.beta(2, 5, size=(n_samples, 4))
    
#     # Create meaningful feature interactions
#     # Example scenario: medical diagnosis based on multiple measurements
#     risk_score = (X[:, 0] * 2.0 +           # Primary risk factor
#                   X[:, 1] * X[:, 2] * 3.0 +  # Interaction effect
#                   np.where(X[:, 3] > 0.7, 1.5, 0) +  # Threshold effect
#                   (X[:, 0] - X[:, 1])**2)    # Discrepancy effect
    
#     threshold = np.percentile(risk_score, 60)  # Top 40% are class 1
#     y = (risk_score > threshold).astype(int)
    
#     return X, y

# X_interactive, y_interactive = interactive_features_4d()

# Generate beta dataset
X, y = generate_normal_dataset(n_samples=9000, alpha=2, beta=3)

# Visualize the distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(X[:, 0], bins=50, alpha=0.7, density=True)
plt.title('Distribution of Feature 1 (Beta)')
plt.xlabel('Value')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('Feature Space')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')

plt.tight_layout()
plt.savefig("./distributions/poisson-5")

save_dataframe(X, y, "poisson-5")



