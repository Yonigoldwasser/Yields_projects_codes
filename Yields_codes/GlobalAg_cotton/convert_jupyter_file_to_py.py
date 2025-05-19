import os
import nbformat
from nbconvert import PythonExporter

# Set the root directory containing your folders
root_dir = r"C:\Users\User\Documents\Yields_project\Yields_codes"

# Iterate through all subdirectories and files
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".ipynb"):
            notebook_path = os.path.join(root, file)
            python_path = os.path.splitext(notebook_path)[0] + ".py"

            # Convert notebook to Python script
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)

            python_exporter = PythonExporter()
            python_code, _ = python_exporter.from_notebook_node(notebook)

            # Write the converted Python script
            with open(python_path, 'w', encoding='utf-8') as f:
                f.write(python_code)

            # Remove the original .ipynb file
            os.remove(notebook_path)
            print(f"Converted and removed: {notebook_path}")

print("\nAll notebooks converted and original files removed successfully.")
