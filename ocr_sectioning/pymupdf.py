import pymupdf # imports the pymupdf library
import os
import json

# Define input and output folders
input_folders = ['folder_1', 'folder_2', 'folder_3', 'folder_4']
output_folders = ['output_1', 'output_2', 'output_3', 'output_4']

# Create output folders if they don't exist
for folder in output_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Create checkpoint file if it doesn't exist
checkpoint_file = 'conversion_checkpoint.json'
completed_files = {}
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        completed_files = json.load(f)

# Process PDFs from each input folder
for input_folder, output_folder in zip(input_folders, output_folders):
    # Get all PDF files in the input folder
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        input_path = os.path.join(input_folder, pdf_file)
        output_path = os.path.join(output_folder, pdf_file.replace('.pdf', '.md'))
        
        # Skip if file was already processed
        checkpoint_key = f"{input_folder}/{pdf_file}"
        if checkpoint_key in completed_files:
            print(f"Skipping {checkpoint_key} - already processed")
            continue
            
        try:
            doc = pymupdf.open(input_path) # open a document
            with open(output_path, 'w', encoding='utf-8') as md_file:
                for page in doc: # iterate the document pages
                    text = page.get_text() # get plain text encoded as UTF-8
                    md_file.write(text + '\n\n')
                doc.close()
                
            # Update checkpoint after successful conversion
            completed_files[checkpoint_key] = True
            with open(checkpoint_file, 'w') as f:
                json.dump(completed_files, f)
                
        except Exception as e:
            print(f"Error processing {checkpoint_key}: {str(e)}")
            continue