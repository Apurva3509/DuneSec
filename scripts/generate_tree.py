import os
from pathlib import Path

def generate_tree(startpath, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = {
            '.git', 
            '__pycache__', 
            'venv', 
            '.pytest_cache', 
            '.ipynb_checkpoints',
            '.conda',
            '.vscode',
            '.idea',
            '__init__.py',
            '.env',
            '.DS_Store'
        }
    
    def print_tree(path, prefix=""):
        entries = [x for x in os.listdir(path) if x not in exclude_dirs and not x.startswith('.')]
        entries = sorted(entries, key=lambda x: (not os.path.isdir(os.path.join(path, x)), x))
        
        tree = []
        for i, entry in enumerate(entries):
            full_path = os.path.join(path, entry)
            is_last = i == len(entries) - 1
            
            if os.path.isdir(full_path):
                # Directory
                tree.append(f"{prefix}{'└──' if is_last else '├──'} {entry}/")
                extension = "    " if is_last else "│   "
                tree.extend(print_tree(full_path, prefix + extension))
            else:
                # File
                tree.append(f"{prefix}{'└──' if is_last else '├──'} {entry}")
        
        return tree

    return "\n".join(["."] + print_tree(startpath))

if __name__ == "__main__":
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Generate the tree
    tree_str = generate_tree(project_root)
    
    # Print to console
    print("\nProject Structure:")
    print("=================")
    print(tree_str)
    print("\n")
    
    # Save to a file
    readme_path = project_root / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the tree section in README
        start_marker = "```\n"
        end_marker = "```"
        tree_start = content.find(start_marker) + len(start_marker)
        tree_end = content.find(end_marker, tree_start)
        
        new_content = (
            content[:tree_start] + 
            tree_str + "\n" + 
            content[tree_end:]
        )
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Tree structure has been updated in README.md")
    else:
        print("README.md not found") 