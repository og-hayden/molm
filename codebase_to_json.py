"""
Script to recursively traverse project directory and output JSON mapping of Python files.
Skips codebase_to_json.py and only includes .py files.
"""
import json
from pathlib import Path
from typing import Dict


def get_python_files(directory: Path) -> Dict[str, str]:
    """
    Recursively find all Python files in directory and read their contents.
    
    Args:
        directory: Root directory to start search from
        
    Returns:
        Dictionary mapping file paths to file contents
    """
    files_dict: Dict[str, str] = {}
    
    for file_path in directory.rglob("*.py"):
        # Skip this script
        if file_path.name == "codebase_to_json.py":
            continue
            
        # Convert path to string and read file contents
        path_str = str(file_path.relative_to(directory))
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                files_dict[path_str] = f.read()
        except Exception as e:
            print(f"Error reading {path_str}: {e}")
            
    return files_dict


def main() -> None:
    """Main function to generate JSON output."""
    # Get project root directory (parent of this script)
    root_dir = Path(__file__).parent
    
    # Get Python files
    files_dict = get_python_files(root_dir)
    
    # Write to JSON file
    output_file = root_dir / "codebase.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(files_dict, f, indent=2)
        
    print(f"Found {len(files_dict)} Python files")
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
