import os
import subprocess
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = os.path.join(os.getcwd(), 'data', 'raw')
PREPROCESSED_DATA_DIR = os.path.join(os.getcwd(), 'data', 'preprocessed')
PREPROCESS_SCRIPT = os.path.join(DATA_DIR, 'leaf', 'data', 'femnist', 'preprocess.sh')
STATS_SCRIPT = os.path.join(DATA_DIR, 'leaf', 'data', 'femnist', 'stats.sh')

# Ensure the directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

# Helper function to run shell commands
def run_command(command, cwd=None):
    try:
        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with error: {e.stderr}")
        raise

# Function to move preprocessed data
def move_preprocessed_data():
    """Moves the preprocessed data to the preprocessed directory."""
    source_train_dir = os.path.join(DATA_DIR, 'leaf', 'data', 'femnist', 'train')
    source_test_dir = os.path.join(DATA_DIR, 'leaf', 'data', 'femnist', 'test')

    if os.path.exists(source_train_dir) and os.path.exists(source_test_dir):
        logger.info("Moving preprocessed train and test data to the preprocessed directory...")
        run_command(['mv', source_train_dir, PREPROCESSED_DATA_DIR])
        run_command(['mv', source_test_dir, PREPROCESSED_DATA_DIR])
    else:
        logger.error(f"Preprocessed data not found in expected directories: {source_train_dir} or {source_test_dir}")
        raise FileNotFoundError(f"Preprocessed data not found in expected directories: {source_train_dir} or {source_test_dir}")

# Function to preprocess the FEMNIST dataset
def preprocess_femnist(iid: bool, sf: float, k: int, t: str, tf: float, smplseed: int = None, spltseed: int = None):
    """Preprocesses the FEMNIST dataset with the specified parameters."""
    logger.info("Preprocessing FEMNIST dataset...")

    # Build the preprocess command
    preprocess_command = [PREPROCESS_SCRIPT]

    if iid:
        preprocess_command += ['-s', 'iid']
    else:
        preprocess_command += ['-s', 'niid']

    preprocess_command += ['--sf', str(sf)]
    preprocess_command += ['-k', str(k)]
    preprocess_command += ['-t', t]
    preprocess_command += ['--tf', str(tf)]

    # Add optional seeds if provided
    if smplseed:
        preprocess_command += ['--smplseed', str(smplseed)]
    if spltseed:
        preprocess_command += ['--spltseed', str(spltseed)]

    # Run the preprocess command
    run_command(preprocess_command, cwd=os.path.join(DATA_DIR, 'leaf/data/femnist'))

    # Move the preprocessed data to the desired location
    move_preprocessed_data()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Download and preprocess the FEMNIST dataset for Federated Learning.")
    
    # Dataset configuration parameters
    parser.add_argument('--iid', action='store_true', help="Set this flag for IID sampling. Default is non-IID.")
    parser.add_argument('--sf', type=float, default=1.0, help="Fraction of data to sample (0.0 < sf <= 1.0). Default is 1.0 (use full dataset).")
    parser.add_argument('-k', type=int, default=0, help="Minimum number of samples per user. Default is 0.")
    parser.add_argument('-t', type=str, choices=['user', 'sample'], default='sample', help="Partitioning method: 'user' or 'sample'. Default is 'sample'.")
    parser.add_argument('--tf', type=float, default=0.9, help="Fraction of data to use for training. Default is 0.9.")
    parser.add_argument('--smplseed', type=int, help="Seed for random sampling (optional).")
    parser.add_argument('--spltseed', type=int, help="Seed for random train-test split (optional).")
    
    args = parser.parse_args()
    
    # Step 1: Download the FEMNIST dataset
    # Assuming the download step is already completed

    # Step 2: Install necessary dependencies
    # Assuming dependencies are already installed

    # Step 3: Preprocess the dataset
    preprocess_femnist(iid=args.iid, sf=args.sf, k=args.k, t=args.t, tf=args.tf, smplseed=args.smplseed, spltseed=args.spltseed)

if __name__ == "__main__":
    main()
