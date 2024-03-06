import logging
import requests
import toml

# Specify the path to your TOML configuration file directly in the module
CONFIG_FILE_PATH = './config.toml'

def is_url_active(url):
    try:
        response = requests.head(url, timeout=5)
        # Check if the response status code is OK (200)
        return response.status_code == 200
    except requests.RequestException as e:
        # Handle exceptions (e.g., timeout, connection error)
        logging.error(f"Error checking URL {url}: {e}")
        return False

# Function to setup logging and load TOML configuration
def setup(log=True, logging_level=logging.INFO):
    # Setup logging
    if log:
        logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load TOML configuration
    with open(CONFIG_FILE_PATH, 'r') as toml_file:
        config = toml.load(toml_file)
    config["url_valid"] = is_url_active(config["base_url"]) # check if LM Studio server is running
    return config
