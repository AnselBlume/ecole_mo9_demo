import importlib
import os
import sys

# Get the directory containing this __init__.py file
package_dir = os.path.dirname(os.path.abspath(__file__))

# Add the package directory to sys.path if it's not already there
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)
print(f"Added {package_dir} to sys.path")
# List of module names to import
modules_to_import = [
    "controller",
    "feature_extraction",
    "image_processing",
    "kb_ops",
    "llm",
    "model",
    "scripts",
    "utils",
    "visualization",
]

# Dictionary to store successfully imported modules
imported_modules = {}

for module_name in modules_to_import:
    try:
        # Try to import the module
        module = importlib.import_module(module_name)
        # If successful, add it to the imported_modules dictionary
        imported_modules[module_name] = module
        # Add it to the global namespace of this __init__.py
        globals()[module_name] = module
    except ImportError as e:
        print(f"Warning: Could not import {module_name}. Error: {e}")

# Set __all__ to the list of successfully imported module names
__all__ = list(imported_modules.keys())

# Print the successfully imported modules (for debugging)
print("Successfully imported modules:", __all__)
