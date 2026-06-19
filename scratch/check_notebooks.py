import json
import sys

def check_notebooks():
    notebooks = [
        '01_EDA_Bank_Marketing.ipynb',
        '02_Modeling_Bank_Marketing.ipynb',
        '03_Advanced_Modeling_And_Imbalance_Handling.ipynb'
    ]
    
    all_clear = True
    for nb_path in notebooks:
        print(f"Checking {nb_path}...")
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
        except Exception as e:
            print(f"  [CRITICAL] Failed to parse JSON: {e}")
            all_clear = False
            continue
            
        errors = 0
        for idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "code":
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "error":
                        ename = output.get("ename", "UnknownError")
                        evalue = output.get("evalue", "")
                        print(f"  [ERROR] in Cell {idx}: {ename} - {evalue}")
                        errors += 1
                        all_clear = False
        if errors == 0:
            print("  Clean! No error outputs found.")
            
    if all_clear:
        print("\nAll notebooks verified and clean!")
        sys.exit(0)
    else:
        print("\nSome notebooks contain errors or failed to parse.")
        sys.exit(1)

if __name__ == "__main__":
    check_notebooks()
