import json
import argparse

def extract_messages(input_file, output_file):
    # Read JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Open output file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Extract messages from the structure
        messages = data.get('messages', [])
        
        for msg in messages:
            # Skip service messages and messages without text
            if msg.get('type') == 'message' and 'text' in msg:
                # Handle both single string and list/object text formats
                text = msg['text']
                if isinstance(text, list):
                    # Concatenate all text parts
                    text_parts = []
                    for part in text:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict) and 'text' in part:
                            text_parts.append(part['text'])
                    text = ' '.join(text_parts)
                
                # Get the username or full name
                from_name = msg.get('from', 'Unknown User')
                
                # Write to file if message is not empty
                if text.strip():
                    f.write(f"{from_name}: {text}\n")

def main():
    parser = argparse.ArgumentParser(description='Extract messages from Telegram JSON export')
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('output_file', help='Path to output TXT file')
    
    args = parser.parse_args()
    extract_messages(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
