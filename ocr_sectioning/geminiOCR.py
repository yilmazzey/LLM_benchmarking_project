import os
import re
import glob
import google.generativeai as genai
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from google.api_core import exceptions
import fitz  # PyMuPDF
import argparse
from tqdm import tqdm

def convert_pdf_to_images(pdf_path, max_pages=None):
    """Convert PDF to a list of images using PyMuPDF."""
    print("Converting PDF to images...")
    # Create output directory if it doesn't exist
    os.makedirs('temp_images', exist_ok=True)
    
    # Open PDF
    pdf_document = fitz.open(pdf_path)
    
    # Determine number of pages to process
    num_pages = min(pdf_document.page_count, max_pages) if max_pages else pdf_document.page_count
    
    # Define footer crop height in pixels - increased to ensure it includes page number
    footer_height = 180  # increased from 120 to 200 pixels to crop from bottom
    
    image_paths = []
    for page_num in range(num_pages):
        # Get page
        page = pdf_document[page_num]
        
        # Convert page to image with higher resolution
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # 3x zoom for better quality
        
        # Create a cropped version of the image (removing footer with page number)
        width, height = pix.width, pix.height
        cropped_height = height - footer_height
        
        # Only crop if there's enough image height
        if cropped_height > height * 0.7:  # Safety check - don't crop more than 30% of the image
            # Create a new PIL image and crop it
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            cropped_img = img.crop((0, 0, width, cropped_height))
            
            # Save image
            image_path = f'temp_images/page_{page_num + 1:03d}.png'
            cropped_img.save(image_path)
        else:
            # If image is too small to safely crop, save the original
            image_path = f'temp_images/page_{page_num + 1:03d}.png'
            pix.save(image_path)
            
        image_paths.append(image_path)
    
    pdf_document.close()
    return image_paths

# Configure Gemini API
def setup_gemini():
    print("Initializing Gemini model...")
    genai.configure(api_key="AIzaSyAFv18kSLlVV4-ClGYCrgaiLtXESJDq5fM")
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

# Retry decorator for rate limit handling
@retry(
    retry=retry_if_exception_type((exceptions.ResourceExhausted, exceptions.ServiceUnavailable)),
    wait=wait_fixed(15),  # Wait 15 seconds between retries
    stop=stop_after_attempt(5)  # Maximum 5 attempts
)
def process_image_with_gemini(model, image_path):
    img = Image.open(image_path)
    response = model.generate_content([
        "Extract all text from this image exactly as it appears, preserving all formatting and line breaks. Do not add any additional text or markers:",
        img
    ])
    return response.text

def process_pdf(pdf_path, max_pages=None):
    # Create output folder if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Generate output filename from PDF name
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    suffix = f"_first_{max_pages}_pages" if max_pages else ""
    output_file = os.path.join('output', f'{pdf_name}{suffix}_extracted.txt')
    
    # Convert PDF to images
    image_paths = convert_pdf_to_images(pdf_path, max_pages)
    
    # Initialize Gemini
    model = setup_gemini()
    
    print(f"\nProcessing {len(image_paths)} pages...")
    
    # Process each image
    with open(output_file, 'w', encoding='utf-8') as f:
        for img_path in tqdm(image_paths, desc="Extracting text"):
            try:
                # Process image with Gemini
                extracted_text = process_image_with_gemini(model, img_path)
                f.write(extracted_text + '\n\n')  # Add double newline between pages
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
                f.write(f"[ERROR: Failed to process this page - {str(e)}]\n\n")
    
    # Clean up temporary images
    for img_path in image_paths:
        try:
            os.remove(img_path)
        except:
            pass
    try:
        os.rmdir('temp_images')
    except:
        pass
    
    print(f"\nText extraction completed!")
    print(f"Text file saved as: {output_file}")
    return output_file

def process_all_pdfs_in_folder(folder_path="article", max_pages=None):
    """Process all PDFs in the specified folder"""
    # Ensure folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        return
    
    # Get all PDF files in the folder
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{folder_path}' folder!")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process:")
    for i, pdf in enumerate(pdf_files):
        print(f"  {i+1}. {os.path.basename(pdf)}")
    
    # Process each PDF file
    for pdf_path in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(pdf_path)}")
        print(f"{'='*60}")
        
        try:
            process_pdf(pdf_path, max_pages)
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
    
    print("\nAll PDF files have been processed!")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Extract text from PDFs using Google Gemini')
    parser.add_argument('--folder', default="articles_100_pdf", help='Folder containing PDF files (default: "articles")')
    parser.add_argument('--max_pages', type=int, help='Maximum number of pages to process per PDF', default=None)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process all PDFs in the folder
    process_all_pdfs_in_folder(args.folder, args.max_pages) 