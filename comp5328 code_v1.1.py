#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load Data Function
def load_data(root='data/CroppedYaleB', reduce=4):
    """ 
    Load ORL (or Extended YaleB) dataset to numpy array.
    
    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.
        
    Returns:
        images: numpy array of shape (num_pixels, num_images).
        labels: numpy array of shape (num_images,).
    """ 
    images, labels = [], []

    for i, person in enumerate(sorted(os.listdir(root))):
        
        if not os.path.isdir(os.path.join(root, person)):
            continue
        
        for fname in os.listdir(os.path.join(root, person)):    
            
            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue
            
            if not fname.endswith('.pgm'):
                continue
                
            # Load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L')  # Convert to grayscale.

            # Reduce computation complexity.
            img = img.resize([s//reduce for s in img.size])

            # Preprocessing: Resize images to a fixed size to ensure consistency.
            fixed_size = (50, 50)  # (width, height)
            img = img.resize(fixed_size)

            # Convert image to numpy array.
            img = np.asarray(img).reshape((-1,1))

            # Collect data and label.
            images.append(img)
            labels.append(i)

    # Concatenate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    return images, labels

# NMF Algorithms

def standard_nmf(V, r, max_iter=100, tol=1e-4):
    """
    Standard NMF algorithm using multiplicative updates with convergence check.

    Objective Function:
        Minimize || V - W H ||_F^2
        where ||.||_F denotes the Frobenius norm.

    Optimization Method:
        Multiplicative Update Rules:
            H <- H * (W^T V) / (W^T W H)
            W <- W * (V H^T) / (W H H^T)
    """
    m, n = V.shape
    W = np.abs(np.random.rand(m, r)) + 1e-9  # Initialize W.
    H = np.abs(np.random.rand(r, n)) + 1e-9  # Initialize H.
    for i in range(max_iter):
        V_prev = W @ H
        # Update H.
        H *= (W.T @ V) / (W.T @ W @ H + 1e-9)  # Numerical stability.
        # Update W.
        W *= (V @ H.T) / (W @ H @ H.T + 1e-9)
        # Compute approximation.
        V_approx = W @ H
        # Check convergence.
        if np.linalg.norm(V_prev - V_approx, 'fro') / np.linalg.norm(V_prev, 'fro') < tol:
            print(f'Standard NMF converged at iteration {i+1}')
            break
    return W, H

def l21_norm_nmf(V, r, max_iter=100, epsilon=1e-9, tol=1e-4):
    """
    Robust NMF using L2,1-norm.

    Objective Function:
        Minimize || V - W H ||_{2,1}
        where ||X||_{2,1} = sum_i || X_{i,:} ||_2

    Optimization Method:
        Iteratively reweighted least squares with weight vector d:
            d_i = 1 / (2 || (V - W H)_{i,:} ||_2 )
        Update W and H by solving weighted least squares problems.
    """
    m, n = V.shape
    W = np.abs(np.random.rand(m, r)) + epsilon  # Initialize W.
    H = np.abs(np.random.rand(r, n)) + epsilon  # Initialize H.
    for i in range(max_iter):
        V_prev = W @ H
        # Compute residuals.
        R = V - V_prev  # (m x n)
        # Compute d vector.
        d = 1 / (2 * np.sqrt(np.sum(R**2, axis=1)) + epsilon)  # (m,)
        # Update W.
        Numerator = (d[:, np.newaxis] * V) @ H.T  # (m x r)
        Denominator = (d[:, np.newaxis] * W @ H) @ H.T + epsilon  # (m x r)
        W *= Numerator / Denominator
        # Update H.
        W_T_D = W.T * d[np.newaxis, :]  # (r x m)
        W_T_D_W = W_T_D @ W  # (r x r)
        W_T_D_V = W_T_D @ V  # (r x n)
        H = np.linalg.solve(W_T_D_W + epsilon * np.eye(r), W_T_D_V)
        H = np.maximum(H, epsilon)  # Ensure non-negativity.
        # Compute approximation.
        V_approx = W @ H
        # Check convergence.
        if np.linalg.norm(V_prev - V_approx, 'fro') / np.linalg.norm(V_prev, 'fro') < tol:
            print(f'L21-Norm NMF converged at iteration {i+1}')
            break
    return W, H

def l1_regularized_nmf(V, r, lambda_param=0.1, max_iter=100, epsilon=1e-9, tol=1e-4):
    """
    L1-Norm Regularized NMF.

    Objective Function:
        Minimize || V - W H ||_F^2 + lambda * || H ||_1

    Optimization Method:
        Multiplicative update rules with regularization to promote sparsity in H.

    Args:
        V: Data matrix (m x n).
        r: Rank of the factorization.
        lambda_param: Regularization parameter controlling sparsity.
        max_iter: Maximum number of iterations.
        epsilon: Small constant to prevent division by zero.
        tol: Tolerance for convergence.

    Returns:
        W: Basis matrix (m x r).
        H: Coefficient matrix (r x n).
    """
    m, n = V.shape
    W = np.abs(np.random.rand(m, r)) + epsilon  # Initialize W.
    H = np.abs(np.random.rand(r, n)) + epsilon  # Initialize H.

    for iteration in range(max_iter):
        V_prev = W @ H

        # Update H
        numerator_H = W.T @ V
        denominator_H = W.T @ W @ H + lambda_param + epsilon
        H *= numerator_H / denominator_H

        # Update W
        numerator_W = V @ H.T
        denominator_W = W @ H @ H.T + epsilon
        W *= numerator_W / denominator_W

        # Compute approximation
        V_approx = W @ H

        # Check convergence
        error = np.linalg.norm(V_prev - V_approx, 'fro') / np.linalg.norm(V_prev, 'fro')
        if error < tol:
            print(f'L1-Regularized NMF converged at iteration {iteration+1}')
            break

    return W, H

# Noise Functions

def add_block_occlusion(image, block_size):
    """Add block occlusion to a given image."""
    img_height, img_width = image.shape
    x = np.random.randint(0, img_width - block_size)
    y = np.random.randint(0, img_height - block_size)
    occluded_image = image.copy()
    occluded_image[y:y+block_size, x:x+block_size] = 255
    return occluded_image

def add_gaussian_noise(image, mean=0, std=25):
    """Add Gaussian noise to a given image."""
    noisy_image = image + np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image

# Relative Reconstruction Error
def relative_reconstruction_error(V_clean, V_approx):
    """Calculate the relative reconstruction error."""
    return np.linalg.norm(V_clean - V_approx) / np.linalg.norm(V_clean)

# Clustering Performance
def evaluate_clustering(H, true_labels):
    """Evaluate clustering performance using KMeans."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, normalized_mutual_info_score

    kmeans = KMeans(n_clusters=len(np.unique(true_labels)), n_init=10, random_state=0).fit(H.T)
    predicted_labels = kmeans.labels_
    accuracy = accuracy_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return accuracy, nmi

# Main Function
if __name__ == "__main__":
    # Adjust the dataset paths accordingly.
    V_orl, Y_orl = load_data(root="C:/Users/fanti/Downloads/data/data/ORL", reduce=4)
    print('ORL Dataset loaded. V shape:', V_orl.shape)
    img_size = (50, 50)  # As specified in the preprocessing.
    image_shape_orl = img_size[::-1]  # (height, width)

    V_yale, Y_yale = load_data(root="C:/Users/fanti/Downloads/data/data/CroppedYaleB", reduce=4)
    print('Yale Dataset loaded. V shape:', V_yale.shape)
    image_shape_yale = img_size[::-1]  # (height, width)

    # List of datasets to process.
    datasets = [
        ('ORL', V_orl, Y_orl, image_shape_orl),
        ('YaleB', V_yale, Y_yale, image_shape_yale)
    ]

    # Block sizes for occlusion.
    block_sizes = [10, 12, 14]  # Different block sizes for experiments.

    for dataset_name, V_data, Y_data, image_shape in datasets:
        print(f"\nProcessing {dataset_name} Dataset:")
        # Choose an image to demonstrate occlusion (e.g., the first image).
        img_index = 0
        img_original = V_data[:, img_index].reshape(image_shape)

        # Apply different block-occlusion noises.
        for b_size in block_sizes:
            img_occluded = add_block_occlusion(img_original, block_size=b_size)

            # Display original and occluded images.
            plt.figure(figsize=(8, 4))
            plt.subplot(121)
            plt.imshow(img_original, cmap='gray')
            plt.title(f'Original Image ({dataset_name})')
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(img_occluded, cmap='gray')
            plt.title(f'Block-Occluded Image b={b_size} ({dataset_name})')
            plt.axis('off')
            plt.show()

            # Replace the occluded image in the dataset.
            V_occluded = V_data.copy()
            V_occluded[:, img_index] = img_occluded.flatten()

            # Run experiments for the occluded image.
            noise_name = f'Block Occlusion b={b_size}'
            print(f"\nProcessing {dataset_name} Dataset with {noise_name}:")

            # Dictionary to store results.
            results = {}

            # List of NMF algorithms to test.
            nmf_algorithms = {
                'Standard NMF': standard_nmf,
                'L21-Norm NMF': l21_norm_nmf,
                'L1-Regularized NMF': l1_regularized_nmf
            }

            for algo_name, nmf_func in nmf_algorithms.items():
                print(f"\nApplying {algo_name}...")
                if algo_name == 'L1-Regularized NMF':
                    W, H = nmf_func(V_occluded, r=40, lambda_param=0.1, max_iter=100)
                else:
                    W, H = nmf_func(V_occluded, r=40, max_iter=100)
                # Reconstruct the image.
                reconstructed = W @ H[:, img_index]
                # Reshape for visualization.
                reconstructed_img = reconstructed.reshape(image_shape)
                # Calculate RRE.
                RRE = relative_reconstruction_error(V_data[:, img_index], reconstructed)
                # Evaluate clustering.
                acc, nmi = evaluate_clustering(H, Y_data)
                # Store results.
                results[algo_name] = {
                    'Reconstructed Image': reconstructed_img,
                    'RRE': RRE,
                    'Accuracy': acc,
                    'NMI': nmi
                }
                print(f"{algo_name} - RRE: {RRE}, Accuracy: {acc}, NMI: {nmi}")

            # Display reconstructed images for comparison.
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(img_occluded, cmap='gray')
            plt.title(f'Occluded Image b={b_size}')
            plt.axis('off')
            for idx, (algo_name, res) in enumerate(results.items(), start=2):
                plt.subplot(1, 4, idx)
                plt.imshow(res['Reconstructed Image'], cmap='gray')
                plt.title(f'{algo_name} Reconstruction')
                plt.axis('off')
            plt.show()

        # Apply Gaussian noise.
        img_noisy = add_gaussian_noise(img_original, std=25)

        # Display original and noisy image.
        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.imshow(img_original, cmap='gray')
        plt.title(f'Original Image ({dataset_name})')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(img_noisy, cmap='gray')
        plt.title(f'Gaussian Noise Image ({dataset_name})')
        plt.axis('off')
        plt.show()

        # Replace the noisy image in the dataset.
        V_noisy = V_data.copy()
        V_noisy[:, img_index] = img_noisy.flatten()

        # Run experiments for the noisy image.
        noise_name = 'Gaussian Noise'
        print(f"\nProcessing {dataset_name} Dataset with {noise_name}:")

        # Dictionary to store results.
        results = {}

        for algo_name, nmf_func in nmf_algorithms.items():
            print(f"\nApplying {algo_name}...")
            if algo_name == 'L1-Regularized NMF':
                W, H = nmf_func(V_noisy, r=40, lambda_param=0.1, max_iter=100)
            else:
                W, H = nmf_func(V_noisy, r=40, max_iter=100)
            # Reconstruct the image.
            reconstructed = W @ H[:, img_index]
            # Reshape for visualization.
            reconstructed_img = reconstructed.reshape(image_shape)
            # Calculate RRE.
            RRE = relative_reconstruction_error(V_data[:, img_index], reconstructed)
            # Evaluate clustering.
            acc, nmi = evaluate_clustering(H, Y_data)
            # Store results.
            results[algo_name] = {
                'Reconstructed Image': reconstructed_img,
                'RRE': RRE,
                'Accuracy': acc,
                'NMI': nmi
            }
            print(f"{algo_name} - RRE: {RRE}, Accuracy: {acc}, NMI: {nmi}")

        # Display reconstructed images for comparison.
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(img_noisy, cmap='gray')
        plt.title('Gaussian Noise Image')
        plt.axis('off')
        for idx, (algo_name, res) in enumerate(results.items(), start=2):
            plt.subplot(1, 4, idx)
            plt.imshow(res['Reconstructed Image'], cmap='gray')
            plt.title(f'{algo_name} Reconstruction')
            plt.axis('off')
        plt.show()


# In[ ]:




