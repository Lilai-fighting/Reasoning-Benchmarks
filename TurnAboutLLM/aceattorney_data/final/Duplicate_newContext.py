import json
import os

def update_contexts():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all JSON files in current directory
    json_files = [f for f in os.listdir(current_dir) if f.endswith('.json')]
    
    for file_name in json_files:
        try:
            file_path = os.path.join(current_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if turns exists
            if 'turns' not in data:
                print(f"{file_name}: No 'turns' array found")
                continue
            
            modified = False
            # Update each turn
            for turn in data['turns']:
                new_context = turn.get('new_context', '')
                old_context = turn.get('newContext', '')
                
                # Keep new_context if both exist
                if new_context and old_context:
                    if new_context != old_context:
                        print(f"{file_name}: Different contents found, keeping new_context")
                    turn['newContext'] = new_context
                    del turn['new_context']
                    modified = True
                # Rename new_context to newContext if it exists
                elif new_context:
                    turn['newContext'] = new_context
                    del turn['new_context']
                    modified = True
                
            # Save changes if any modifications were made
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                print(f"{file_name}: Updated context fields")
                    
        except json.JSONDecodeError:
            print(f"{file_name}: Invalid JSON file")
        except Exception as e:
            print(f"{file_name}: Error - {str(e)}")

if __name__ == "__main__":
    update_contexts() 