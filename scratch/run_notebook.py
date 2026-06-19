import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys

def execute_nb():
    notebook_path = "01_EDA_Bank_Marketing.ipynb"
    print(f"Reading notebook {notebook_path}...")
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        
        print("Executing notebook cells. This may take a few moments...")
        # Configure the execution preprocessor
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        
        # Execute the notebook in the current working directory
        ep.preprocess(nb, {"metadata": {"path": "."}})
        
        print("Saving executed notebook...")
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
            
        print("Notebook executed and saved successfully!")
    except Exception as e:
        print(f"Error executing notebook: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    execute_nb()
