import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_folder", type=str, required=True, help="Folder containing the transcript JSON file")
    parser.add_argument("--output_file", type=str, default="last_100_transcripts.json", help="Output JSON file path")
    return parser.parse_args()

def main(args):
    # Find transcript file
    for file in os.listdir(args.test_folder):
        if file.endswith(".json"):
            transcript_file = os.path.join(args.test_folder, file)
            break
    
    # Load transcript data
    with open(transcript_file, 'r', encoding='utf-8') as f:
        print(f"Loading transcripts from {transcript_file}")
        transcript_data = json.load(f)
    
    print(f"Total entries in transcript data: {len(transcript_data)}")
    
    # Determine if transcript_data is a list or a dictionary
    if isinstance(transcript_data, list):
        # If it's a list, take the last 100 items
        last_100_entries = transcript_data[-100:] if len(transcript_data) > 100 else transcript_data
    else:
        # If it's a dictionary, convert to items, take last 100, convert back to dict
        items = list(transcript_data.items())
        last_100_items = items[-100:] if len(items) > 100 else items
        last_100_entries = dict(last_100_items)
    
    num_entries = len(last_100_entries)
    print(f"Extracted last {num_entries} entries")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the last 100 entries to the new JSON file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(last_100_entries, f, ensure_ascii=False, indent=4)
    
    print(f"Saved {num_entries} entries to {args.output_file}")

if __name__ == "__main__":
    args = parse_args()
    main(args) 