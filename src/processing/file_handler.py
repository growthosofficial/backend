from typing import Optional

def read_input_from_file(file_path: str) -> Optional[str]:
    """
    Read input text from a .txt file
    
    Args:
        file_path: Path to the .txt file
        
    Returns:
        String content of the file or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        print(f"📄 Read {len(content)} characters from {file_path}")
        return content
    
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None