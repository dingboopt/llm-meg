import json
import sys

# Function to remove specified fields from each item in the JSON data
def modify_data(json_data):
    for item in json_data:
        for field in ['semantic']:
            if field in item:
                del item[field]
        if item["wav_path"].startswith('/cloudfs'):
            item["wav_path"]='/workspace'+item["wav_path"][len('/cloudfs'):]
            #print(item["wav_path"])

# Load the JSON file
input_file = sys.argv[1]  # Replace with your file path
output_file = sys.argv[2]  # The file where modified JSON will be saved

with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)


# Remove specified fields
modify_data(data)

# Save the modified JSON back to a file
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False,indent=4)

print(f"Modified JSON saved to {output_file}")
