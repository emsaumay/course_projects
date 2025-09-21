# Fraud Assignmnet 4
# Saumay Lunawat CS24MTECH14005
# Shivendra Deshpande CS24MTECH12017
# Chinmay Rajesh Ingle CS23MTECH12002

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set default matplotlib style and font for better readability
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'font.size': 10})

# 1. Generator Network
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim, hidden_dims=[128, 256, 128]):
        super(Generator, self).__init__()

        layers = []

        # First layer from noise to first hidden layer
        layers.append(nn.Linear(noise_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))

        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # Removed Tanh activation
        # layers.append(nn.Tanh()) # Tanh restricts output to [-1, 1] which conflicts with StandardScaler

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)

# 2. Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super(Discriminator, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))

        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))

        # Output layer - single value for real/fake classification
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid()) # Sigmoid outputs in (0, 1) for BCELoss

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 3. Training Function
def train_gan(generator, discriminator, dataloader, noise_dim, num_epochs=500,
              lr_g=0.0002, lr_d=0.0002, beta1=0.5, beta2=0.999, device='cpu'):

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

    # Labels for real and fake data
    # Smoothed labels often improve GAN training stability
    real_label = 0.9 # Use slightly less than 1.0 for real labels
    fake_label = 0.1 # Use slightly more than 0.0 for fake labels

    # Training history
    g_losses = []
    d_losses = []

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        batches = 0

        for batch_data in dataloader:
            batch_size = batch_data[0].size(0)
            batches += 1

            # -----------------------------
            # Train Discriminator
            # -----------------------------

            # Train with real data
            optimizer_d.zero_grad()
            real_data = batch_data[0].to(device)
            # Use smoothed labels
            label = torch.full((batch_size, 1), real_label, dtype=torch.float, device=device)

            output = discriminator(real_data)
            d_loss_real = criterion(output, label)
            d_loss_real.backward()

            # Train with fake data
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_data = generator(noise)
            # Use smoothed labels
            label.fill_(fake_label)

            # Detach generator output to prevent gradients from flowing to the generator during discriminator training
            output = discriminator(fake_data.detach())
            d_loss_fake = criterion(output, label)
            d_loss_fake.backward()

            # Update discriminator - Gradients from both real and fake batches are accumulated
            d_loss = d_loss_real + d_loss_fake
            optimizer_d.step()

            # -----------------------------
            # Train Generator
            # -----------------------------
            optimizer_g.zero_grad()
            # Generator wants discriminator to think its output is real, use real_label
            label.fill_(real_label)

            output = discriminator(fake_data) # Use fake_data without detaching
            g_loss = criterion(output, label)
            g_loss.backward()

            # Update generator
            optimizer_g.step()

            # Accumulate losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        # Average losses over batches
        epoch_d_loss /= batches
        epoch_g_loss /= batches

        # Store losses for plotting
        g_losses.append(epoch_g_loss)
        d_losses.append(epoch_d_loss)

        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 1: # Print first and every 50th epoch
            print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {epoch_d_loss:.4f} | G Loss: {epoch_g_loss:.4f}")

    print("Training finished.")
    return g_losses, d_losses

# 4. Generate Synthetic Data
def generate_synthetic_data(generator, n_samples, noise_dim, device='cpu'):
    generator.eval() # Set generator to evaluation mode
    synthetic_data = []
    batch_size = 1024 # Generate in batches to avoid memory issues for large n_samples
    with torch.no_grad():
        for _ in range(n_samples // batch_size + 1):
            current_batch_size = min(batch_size, n_samples - len(synthetic_data))
            if current_batch_size == 0:
                break
            noise = torch.randn(current_batch_size, noise_dim, device=device)
            fake_data = generator(noise).cpu().numpy()
            synthetic_data.append(fake_data)
    synthetic_data = np.concatenate(synthetic_data, axis=0)
    return synthetic_data

# 5. Evaluate Synthetic Data
def evaluate_synthetic_data(real_data, synthetic_data, feature_names=None):
    """Compare distributions and correlations between real and synthetic data"""

    print("\n--- Evaluating Synthetic Data Quality ---")

    # If feature names not provided, create generic ones
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(real_data.shape[1])]

    # Create DataFrames
    real_df = pd.DataFrame(real_data, columns=feature_names)
    synthetic_df = pd.DataFrame(synthetic_data, columns=feature_names)

    # 1. Plot distributions for each feature
    print("Generating feature distribution plots...")
    n_features = real_data.shape[1]
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 5, n_rows * 4)) # Adjusted figure size

    for i, feature in enumerate(feature_names):
        plt.subplot(n_rows, n_cols, i+1)

        # Determine appropriate number of bins
        # Use Freedman-Diaconis rule or a reasonable default
        try:
            # Combine data to estimate bins across both real and synthetic
            combined_data = np.concatenate([real_df[feature].dropna(), synthetic_df[feature].dropna()])
            if len(combined_data) > 0:
                 bins = int(np.ceil((np.max(combined_data) - np.min(combined_data)) / (2 * (np.percentile(combined_data, 75) - np.percentile(combined_data, 25)) / (len(combined_data)**(1/3)))))
                 bins = max(10, min(bins, 100)) # Clamp bins between 10 and 100
            else:
                 bins = 30 # Default if no data

        except Exception:
             bins = 30 # Default on error


        plt.hist(real_df[feature], alpha=0.6, bins=bins, label='Real', density=True) # Increased alpha slightly
        plt.hist(synthetic_df[feature], alpha=0.6, bins=bins, label='Synthetic', density=True) # Increased alpha slightly
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()

    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()
    print("Feature distribution plots saved to 'feature_distributions.png'")

    # 2. Compute and plot correlation matrices
    print("Generating correlation matrix comparison plots...")
    real_corr = real_df.corr()
    synthetic_corr = synthetic_df.corr()

    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)
    sns.heatmap(real_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f") # Added format
    plt.title('Correlation Matrix - Real Data')

    plt.subplot(1, 2, 2)
    sns.heatmap(synthetic_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f") # Added format
    plt.title('Correlation Matrix - Synthetic Data')

    plt.tight_layout()
    plt.savefig('correlation_comparison.png')
    plt.close()
    print("Correlation matrices comparison saved to 'correlation_comparison.png'")

    # 3. Calculate correlation difference
    print("Generating correlation difference plot...")
    # Calculate absolute difference and then mean absolute difference
    corr_diff_abs = np.abs(real_corr - synthetic_corr)
    # Exclude diagonal (self-correlation is always 1) and duplicate lower/upper triangle
    mask = np.triu(np.ones_like(corr_diff_abs, dtype=bool))
    corr_diff_abs_masked = corr_diff_abs.mask(mask)
    mean_corr_diff = corr_diff_abs_masked.mean().mean()


    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_diff_abs, annot=True, cmap='Reds', fmt=".2f", vmin=0, vmax=1) # Set vmin/vmax
    plt.title(f'Absolute Correlation Difference (Mean Abs Diff: {mean_corr_diff:.4f})')
    plt.tight_layout()
    plt.savefig('correlation_difference.png')
    plt.close()
    print("Correlation difference heatmap saved to 'correlation_difference.png'")

    # 4. Calculate additional statistics
    print("Calculating mean and standard deviation comparison...")
    real_mean = np.mean(real_data, axis=0)
    synth_mean = np.mean(synthetic_data, axis=0)
    real_std = np.std(real_data, axis=0)
    synth_std = np.std(synthetic_data, axis=0)

    # Mean and standard deviation comparison
    stats_comparison = pd.DataFrame({
        'Feature': feature_names,
        'Real_Mean': real_mean,
        'Synthetic_Mean': synth_mean,
        'Mean_Diff': np.abs(real_mean - synth_mean),
        'Real_StdDev': real_std,
        'Synthetic_StdDev': synth_std,
        'StdDev_Diff': np.abs(real_std - synth_std)
    })

    stats_comparison.to_csv('statistics_comparison.csv', index=False)
    print("Statistical comparison saved to 'statistics_comparison.csv'")

    print("--- Evaluation Complete ---")

    return mean_corr_diff, stats_comparison


# 6. Main function to run the entire process
def main():
    # Load data from Excel file
    print("Loading data from data.xlsx...")
    try:
        # Specify engine for potentially different Excel formats
        df = pd.read_excel('data.xlsx', engine='openpyxl')
        print(f"Successfully loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print("Error: data.xlsx not found. Please make sure the file is in the same directory.")
        return
    except ImportError:
         print("Error: openpyxl library not found. Please install it: pip install openpyxl")
         return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Extract feature names and data
    feature_names = df.columns.tolist()
    X = df.values

    # Handle any missing values BEFORE scaling
    # Common strategies: mean, median, or a constant
    if np.isnan(X).any():
        print("Warning: Data contains missing values. Filling with median values.")
        try:
            # Calculate median ignoring NaNs
            col_median = np.nanmedian(X, axis=0)
            # Use the median values to fill NaNs column-wise
            for i in range(X.shape[1]):
                 if np.isnan(X[:, i]).any():
                      X[np.isnan(X[:, i]), i] = col_median[i]
            print("Missing values filled with column medians.")
        except Exception as e:
             print(f"Error during median imputation: {e}. Exiting.")
             return

    # Preprocess data - scale using StandardScaler
    print("\nScaling data with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaled.")

    X_train_scaled = X_scaled # Train on the entire scaled dataset

    # Convert to torch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    dataset = TensorDataset(X_train_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=0) # Increased batch size, added pin_memory, num_workers=0 for simplicity

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Initialize models
    input_dim = X_train_scaled.shape[1]
    noise_dim = 100

    generator = Generator(noise_dim=noise_dim, output_dim=input_dim).to(device)
    discriminator = Discriminator(input_dim=input_dim).to(device)

    # Print model summary
    print("\nGenerator Architecture:")
    print(generator)
    print("\nDiscriminator Architecture:")
    print(discriminator)

    # Train GAN
    print("\nStarting GAN training...")
    g_losses, d_losses = train_gan(
        generator,
        discriminator,
        dataloader,
        noise_dim=noise_dim,
        num_epochs=1000,  # Increased epochs for potentially better results
        lr_g=0.0002,
        lr_d=0.0002,
        beta1=0.5,
        beta2=0.999,
        device=device
    )

    # Plot training losses
    print("\nPlotting training losses...")
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    plt.grid(True)
    plt.savefig('gan_training_losses.png')
    plt.close()
    print("Training loss plot saved to 'gan_training_losses.png'")

    # Generate synthetic data
    n_samples = len(X)  # Generate same number of samples as original dataset
    print(f"\nGenerating {n_samples} synthetic samples...")
    synthetic_data_scaled = generate_synthetic_data(generator, n_samples, noise_dim, device)
    print("Synthetic data generated (scaled).")

    # Inverse transform the data back to original scale
    print("Inverse transforming synthetic data...")
    synthetic_data_original = scaler.inverse_transform(synthetic_data_scaled)
    X_original = scaler.inverse_transform(X_scaled) # Get original data back for comparison

    print("Inverse transformation complete.")

    # Evaluate synthetic data
    mean_corr_diff, stats_comparison = evaluate_synthetic_data(X_original, synthetic_data_original, feature_names)
    print(f"\nFinal Mean absolute correlation difference: {mean_corr_diff:.4f}")

    # Save synthetic data
    synthetic_df = pd.DataFrame(synthetic_data_original, columns=feature_names)
    synthetic_df.to_csv('synthetic_data.csv', index=False)
    print("\nSynthetic data saved to 'synthetic_data.csv'")

    print("\nProcess finished.")

if __name__ == "__main__":
    main()