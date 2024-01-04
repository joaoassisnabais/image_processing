import os

def parse_config_file (relative_file_path):
    """Parse a configuration file and return a dictionary of parameters and values."""
    
    file_path = os.path.join(os.path.dirname(__file__), '..', relative_file_path)
    config_dict = {} 
    
    with open(file_path, 'r') as file:
        for line in file: 
            line = line.strip()
            
            # Ignore comments
            if line.startswith('#') or line == '':
                continue
            
            # Split the line into tokens
            tokens = line.split()
            
            # Extract parameter names and values 
            param_name = tokens[0] 
            param_values = [tokens[1:]] 
            
            # Check if the token already exists in the dictionary 
            if param_name in config_dict: 
                # Add new values to the existing token
                config_dict[param_name].extend(param_values)
            else: # Create a new entry in the dictionary 
                config_dict[param_name] = param_values
    
    return config_dict

def convert_file_path(file_path):
    """Convert a file path to the correct format for the current OS."""
    if os.name == 'nt':  # 'nt' represents Windows OS
        absolute_path = os.path.dirname(__file__)
        absolute_path = os.path.join(absolute_path, "..", file_path)
        return absolute_path.replace("/", "\\")

