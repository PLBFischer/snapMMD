import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import logging
from snapMMD.dls import MMDLoss, RBF
from scipy.optimize import linprog
from sklearn.decomposition import PCA

# Dataset configurations
DATASETS = {
    'LV': {
        'data_path': 'data/classic/LV_data.npz',
        'dimensionality': 2,
        'axes_labels': ['Prey', 'Predator'],
        'title': 'Lotka-Volterra',
        'calculate_emd': True
    },
    'Repressilator': {
        'data_path': 'data/classic/Repressilator_data.npz',
        'dimensionality': 3,
        'axes_labels': ['Gene 1', 'Gene 2', 'Gene 3'],
        'title': 'Repressilator',
        'calculate_emd': True
    },
    'GoM': {
        'data_path': 'data/realdata/GoM_data.npz',
        'dimensionality': 2,
        'axes_labels': ['X1', 'X2'],
        'title': 'GoM',
        'calculate_emd': True
    },
    'pbmc': {
        'data_path': 'data/realdata/processed_pbmc_data_sub500_every_2_until20.npz',
        'dimensionality': 30,
        'plot_dimensionality': 3,
        'axes_labels': ['PC1', 'PC2', 'PC3'],
        'title': 'PBMC',
        'calculate_emd': False,  # doesn't converge
        'requires_pca': True
    }
}

def setup_pca_for_pbmc(logger):
    """Set up PCA for PBMC dataset."""
    # Load both PBMC datasets for PCA computation
    data1 = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    data2 = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20_interp_val.npz")
    
    Xs1, Xs2 = data1["Xs"], data2["Xs"]
    
    # Combine datasets and fit PCA
    Xs_combined = np.concatenate([Xs1, Xs2], axis=0)
    n_timepoints, n_cells, n_genes = Xs_combined.shape
    X_reshaped = Xs_combined.reshape(n_timepoints * n_cells, n_genes)
    
    pca = PCA(n_components=3)
    pca.fit(X_reshaped)
    
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return pca

def load_data(dataset_name, seed):
    """Load training data and forecast results."""
    config = DATASETS[dataset_name]
    
    # Load training data
    data = np.load(config['data_path'])
    N_steps = data['N_steps']
    Xs_training = [data["Xs"][i] for i in range(N_steps-1)]
    X_val_true = data["Xs"][-1]
    
    # Load forecast results
    forecast_data = np.load(f"forecasts/{dataset_name}_forecast_{seed}.npz")
    forecast = forecast_data['forecast'][1:]  # Take second element only
    
    return Xs_training, X_val_true, forecast, config

def plot_results(dataset_name, seed, Xs_training, X_val_true, forecast, config, output_dir, logger, pca=None):
    """Plot training data, ground truth, and forecast."""
    
    # Transform data if PCA is needed
    def transform_data(data):
        if pca is not None:
            return pca.transform(data)
        return data
    
    # Set up figure
    is_3d = config.get('plot_dimensionality', config['dimensionality']) == 3
    if is_3d:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot training sequences with color progression
    n_sequences = len(Xs_training)
    cmap = plt.colormaps.get_cmap('coolwarm')
    
    all_training_data = []
    timepoint_colors = []
    
    for i, X in enumerate(Xs_training):
        X_plot = transform_data(X)
        all_training_data.append(X_plot)
        timepoint_colors.extend([i+1] * len(X_plot))
    
    all_training_data = np.concatenate(all_training_data, axis=0)
    
    # Create scatter plot with colorbar
    if is_3d:
        scatter = ax.scatter(all_training_data[:, 0], all_training_data[:, 1], all_training_data[:, 2], 
                          alpha=0.7, s=3.0, c=timepoint_colors, cmap=cmap, vmin=1, vmax=n_sequences)
        ax.set_zlabel(config['axes_labels'][2])
        if dataset_name == 'Repressilator':  # Front view for Repressilator
            ax.view_init(elev=20, azim=45)
    else:
        scatter = ax.scatter(all_training_data[:, 0], all_training_data[:, 1], 
                          alpha=0.7, s=3.0, c=timepoint_colors, cmap=cmap, vmin=1, vmax=n_sequences)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Time Point', rotation=270, labelpad=15)
    
    # Plot ground truth and forecast
    true_plot = transform_data(X_val_true)
    forecast_plot = transform_data(forecast[-1])
    
    if is_3d:
        ax.scatter(true_plot[:, 0], true_plot[:, 1], true_plot[:, 2],
                  alpha=0.9, s=8.0, color='darkgreen', label='Ground Truth', 
                  marker='o', edgecolor='white', linewidth=0.5)
        ax.scatter(forecast_plot[:, 0], forecast_plot[:, 1], forecast_plot[:, 2],
                  alpha=0.9, s=8.0, color='darkorange', label='Forecast',
                  marker='s', edgecolor='white', linewidth=0.5)
    else:
        ax.scatter(true_plot[:, 0], true_plot[:, 1],
                  alpha=0.9, s=8.0, color='darkgreen', label='Ground Truth',
                  marker='o', edgecolor='white', linewidth=0.5)
        ax.scatter(forecast_plot[:, 0], forecast_plot[:, 1],
                  alpha=0.9, s=8.0, color='darkorange', label='Forecast',
                  marker='s', edgecolor='white', linewidth=0.5)
    
    # Labels and formatting
    ax.set_xlabel(config['axes_labels'][0])
    ax.set_ylabel(config['axes_labels'][1])
    ax.set_title('Training Data, Ground Truth & Forecast Phase Portrait')
    ax.legend()
    if not is_3d:
        ax.grid(True)
    
    # Title and save
    title = f"{config['title']} Results (Seed {seed})"
    fig.suptitle(title)
    
    filename = f"{dataset_name}_results_seed_{seed}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot saved: {filepath}")

def calculate_emd(x, y):
    """Calculate Earth Mover Distance"""
    n, m = x.shape[0], y.shape[0]
    
    C = np.linalg.norm(x[:,None] - y[None,:], axis=2).ravel()
    
    A_eq = []
    b_eq = []
    
    # Row constraints
    for i in range(n):
        row = np.zeros(n*m)
        row[i*m:(i+1)*m] = 1
        A_eq.append(row)
        b_eq.append(1/n)
    
    # Column constraints
    for j in range(m):
        row = np.zeros(n*m)
        row[j::m] = 1
        A_eq.append(row)
        b_eq.append(1/m)
    
    res = linprog(C, A_eq=np.vstack(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method='highs')
    
    if res.success:
        return res.fun
    else:
        return np.nan

def calculate_scores(dataset_name, seeds, logger):
    """Calculate MMD and EMD scores for multiple seeds."""
    config = DATASETS[dataset_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up MMD
    rbf = RBF(bandwidth=1.0).to(device)
    mmd_loss = MMDLoss(kernel=rbf).to(device)
    
    # Load ground truth
    data = np.load(config['data_path'])
    X_val = torch.tensor(data["Xs"][-1]).to(device)
    X_val_np = data["Xs"][-1]
    
    mmd_scores = []
    emd_scores = []
    successful_seeds = []
    
    logger.info(f"\nCalculating scores for {dataset_name}:")
    logger.info("-" * 50)
    
    for seed in seeds:
        try:
            forecast_data = np.load(f"forecasts/{dataset_name}_forecast_{seed}.npz")
            forecast = forecast_data['forecast'][1:]  # Take second element
            forecast_final = forecast[-1]
            
            # Calculate MMD
            forecast_tensor = torch.tensor(forecast_final).to(device)
            mmd_squared = mmd_loss(forecast_tensor, X_val)
            mmd = np.sqrt(mmd_squared.item())
            mmd_scores.append(mmd)
            
            # Calculate EMD if enabled
            if config['calculate_emd']:
                emd = calculate_emd(forecast_final, X_val_np)
                emd_scores.append(emd)
                logger.info(f"Seed {seed}: MMD = {mmd:.6f}, EMD = {emd:.6f}")
            else:
                logger.info(f"Seed {seed}: MMD = {mmd:.6f}")
            
            successful_seeds.append(seed)
            
        except FileNotFoundError:
            logger.warning(f"Seed {seed}: File not found, skipping")
    
    # Calculate statistics
    if mmd_scores:
        mmd_array = np.array(mmd_scores)
        logger.info(f"\nMMD Results:")
        logger.info(f"Mean ± Std: {np.mean(mmd_array):.6f} ± {np.std(mmd_array):.6f}")
        
        if emd_scores and config['calculate_emd']:
            emd_array = np.array(emd_scores)
            logger.info(f"\nEMD Results:")
            logger.info(f"Mean ± Std: {np.mean(emd_array):.6f} ± {np.std(emd_array):.6f}")
    
    return successful_seeds, mmd_scores, emd_scores

def setup_logging(dataset_name, output_dir):
    """Set up logging to file."""
    log_filename = f"{dataset_name}_analysis.log"
    log_filepath = os.path.join(output_dir, log_filename)
    
    # Set up logger
    logger = logging.getLogger(f'{dataset_name}_analysis')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger, log_filepath

def main():
    parser = argparse.ArgumentParser(description='Analyze forecasting results')
    parser.add_argument('--dataset', type=str, choices=list(DATASETS.keys()), 
                       required=True, help='Dataset name')
    parser.add_argument('--seed', type=int, default=42, help='Seed for plotting (default: 42)')
    parser.add_argument('--score-seeds', nargs='+', type=int, 
                       default=[1, 2, 3, 4, 5, 40, 41, 42, 43, 44],
                       help='Seeds for scoring (default: 1 2 3 4 5 40 41 42 43 44)')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                       help='Output directory (default: analysis_results)')
    
    args = parser.parse_args()
    
    # Create output directory
    dataset_output_dir = os.path.join(args.output_dir, f"{args.dataset}_analysis")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Set up logging
    logger, log_filepath = setup_logging(args.dataset, dataset_output_dir)
    
    logger.info(f"Starting analysis for {args.dataset} dataset")
    logger.info(f"Plot seed: {args.seed}")
    logger.info(f"Score seeds: {args.score_seeds}")
    logger.info(f"Output directory: {dataset_output_dir}")
    
    # Set up PCA for PBMC if needed
    pca = None
    if DATASETS[args.dataset].get('requires_pca', False):
        logger.info("Setting up PCA for PBMC dataset...")
        pca = setup_pca_for_pbmc(logger)
    
    # Load data and create plot
    logger.info("Loading data and creating plot...")
    Xs_training, X_val_true, forecast, config = load_data(args.dataset, args.seed)
    plot_results(args.dataset, args.seed, Xs_training, X_val_true, forecast, config, 
                dataset_output_dir, logger, pca)
    
    # Calculate scores across multiple seeds
    logger.info("Calculating scores across multiple seeds...")
    calculate_scores(args.dataset, args.score_seeds, logger)
    
    logger.info(f"Analysis complete for {args.dataset}!")
    logger.info(f"Results saved to: {dataset_output_dir}")
    logger.info(f"Log file: {log_filepath}")
    
    print(f"Analysis complete! Results saved to: {dataset_output_dir}")
    print(f"Log file: {log_filepath}")

if __name__ == "__main__":
    main()
