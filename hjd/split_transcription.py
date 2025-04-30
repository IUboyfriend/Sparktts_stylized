import os
import json
import argparse

def split_transcription_file(json_path, output_dir, train_count=792):
    """
    Split a transcription_results.json file into training (first 792 entries) 
    and testing (remaining entries) sets.
    
    Args:
        json_path: Path to transcription_results.json
        output_dir: Directory to output the split JSON files
        train_count: Number of entries to include in training set (default: 792)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load transcription data
    with open(json_path, 'r') as f:
        transcription_data = json.load(f)
    
    # Get all filenames in order
    all_filenames = list(transcription_data.keys())
    
    # Split into training and testing sets
    train_filenames = all_filenames[:train_count]
    test_filenames = all_filenames[train_count:]
    
    # Create the split dictionaries
    train_transcriptions = {filename: transcription_data[filename] for filename in train_filenames}
    test_transcriptions = {filename: transcription_data[filename] for filename in test_filenames}
    
    # Save split transcription files
    train_path = os.path.join(output_dir, 'train_transcription_results.json')
    test_path = os.path.join(output_dir, 'test_transcription_results.json')
    
    with open(train_path, 'w') as f:
        json.dump(train_transcriptions, f, indent=2)
    
    with open(test_path, 'w') as f:
        json.dump(test_transcriptions, f, indent=2)
    
    # Print summary and filenames
    print(f"Dataset split complete:")
    print(f"  Total entries: {len(all_filenames)}")
    print(f"  Training entries: {len(train_filenames)}")
    print(f"  Testing entries: {len(test_filenames)}")
    
    print("\nTraining filenames:")
    for filename in train_filenames:
        print(f"  {filename}")
    
    print("\nTesting filenames:")
    for filename in test_filenames:
        print(f"  {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split transcription_results.json into training and testing sets")
    parser.add_argument("--json_path", type=str, required=True, help="Path to transcription_results.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to output the split JSON files")
    parser.add_argument("--train_count", type=int, default=792, help="Number of entries for training set")
    
    args = parser.parse_args()
    
    split_transcription_file(args.json_path, args.output_dir, args.train_count)






   