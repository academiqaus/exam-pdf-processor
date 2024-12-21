import streamlit as st
import os
import PyPDF2
import fitz
import base64
from werkzeug.utils import secure_filename
import pandas as pd
import logging
import sys
import io
import shutil
import time
from pathlib import Path
from openai import OpenAI, AuthenticationError
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from canvasapi import Canvas
import difflib
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_FOLDER = '/tmp/uploads' if os.getenv('STREAMLIT_SERVER_HEADLESS') == 'true' else 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
DUPLICATE_FOLDER = os.path.join(UPLOAD_FOLDER, 'duplicate_splits')
MODEL = "gpt-4o-mini"  # Updated to use the correct model

# Global variable for OpenAI client
client = None

def cleanup_old_files():
    """Clean up files older than 1 hour"""
    try:
        current_time = time.time()
        for root, dirs, files in os.walk(UPLOAD_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                # If file is older than 1 hour, delete it
                if current_time - os.path.getmtime(file_path) > 3600:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Error deleting old file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def create_upload_folders():
    """Create necessary folders for file uploads"""
    folders = [
        UPLOAD_FOLDER,
        os.path.join(UPLOAD_FOLDER, 'splits'),
        os.path.join(UPLOAD_FOLDER, 'preview'),
        DUPLICATE_FOLDER
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    # Clean up old files
    cleanup_old_files()

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(uploaded_file, save_path):
    """Save an uploaded file to the specified path"""
    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False

# Function to encode an image as base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to extract the top third of the first page of a PDF as a PNG
def extract_top_third_as_png(pdf_path, zoom=2):
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    rect = page.rect
    top_third = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1 / 3)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=top_third)
    png_path = pdf_path.replace(".pdf", "_top_third.png")
    pix.save(png_path)
    return png_path

# Process a single PDF with OpenAI
def process_pdf_with_ai(pdf_path, max_retries=3):
    global client
    try:
        png_path = extract_top_third_as_png(pdf_path, zoom=2)
        base64_image = encode_image(png_path)

        messages = [
            {"role": "system", "content": "You are an expert exam file processor that extracts handwritten student numbers and names."},
            {"role": "user", "content": [
                {"type": "text", "text": "What is the student number and student name in this image? Please output as follows:\nNAME: <OUTPUT>\nNUMBER: <OUTPUT>"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ]

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.0,
                )
                extracted_text = response.choices[0].message.content

                # Extract student info
                student_number_match = re.search(r"NUMBER: (\d+)", extracted_text)
                student_name_match = re.search(r"NAME: (.+)", extracted_text)

                student_number = student_number_match.group(1) if student_number_match else "UnknownNumber"
                student_name = student_name_match.group(1).strip() if student_name_match else "UnknownName"
                sanitized_student_name = student_name.replace(" ", "_")

                os.remove(png_path)  # Clean up PNG
                return student_number, sanitized_student_name, extracted_text

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                continue

    except Exception as e:
        if os.path.exists(png_path):
            os.remove(png_path)
        raise e

def process_files_with_openai(files, session_folder, api_key):
    """Process multiple files with OpenAI in parallel"""
    global client
    if client is None:
        client = OpenAI(api_key=api_key)
    
    processed_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_file = {
            executor.submit(process_pdf_with_ai, 
                          os.path.join(session_folder, filename)): filename 
            for filename in files
        }
        
        # Create a progress bar
        progress_bar = st.progress(0)
        total_files = len(files)
        completed = 0
        
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                student_number, student_name, extracted_text = future.result()
                new_filename = f"{student_number}_{student_name}.pdf"
                
                # Handle duplicates
                old_path = os.path.join(session_folder, filename)
                new_path = os.path.join(session_folder, new_filename)
                duplicate_path = os.path.join(DUPLICATE_FOLDER, new_filename)
                
                count = 1
                while os.path.exists(new_path) or os.path.exists(duplicate_path):
                    new_filename = f"{student_number}_{student_name}_{count}.pdf"
                    new_path = os.path.join(session_folder, new_filename)
                    duplicate_path = os.path.join(DUPLICATE_FOLDER, new_filename)
                    count += 1

                # Rename and copy
                os.rename(old_path, new_path)
                shutil.copy2(new_path, duplicate_path)
                
                processed_results.append({
                    'original_filename': filename,
                    'new_filename': new_filename,
                    'extracted_text': extracted_text,
                    'success': True
                })
                
            except Exception as e:
                processed_results.append({
                    'original_filename': filename,
                    'error': str(e),
                    'success': False
                })
            
            # Update progress
            completed += 1
            progress_bar.progress(completed / total_files)
    
    return processed_results

def extract_course_assignment_ids(assignment_url):
    """Extracts course_id, assignment_id, and base URL from the Canvas assignment URL."""
    try:
        # Extract base URL (everything up to /courses)
        base_url = assignment_url[:assignment_url.find('/courses')]
        # Extract IDs
        parts = assignment_url.strip().split('/')
        course_id = parts[4]
        assignment_id = parts[6]
        return base_url, course_id, assignment_id
    except IndexError:
        st.error("Invalid assignment URL format")
        return None, None, None

def authenticate_canvas(api_url, api_key):
    """Authenticates with the Canvas API and returns the Canvas object."""
    try:
        canvas = Canvas(api_url, api_key)
        return canvas
    except Exception as e:
        st.error(f"Failed to authenticate with Canvas API: {str(e)}")
        return None

def get_course_students(canvas, course_id):
    """Retrieves all students from a course."""
    try:
        course = canvas.get_course(course_id)
        return list(course.get_users(enrollment_type=['student']))
    except Exception as e:
        st.error(f"Failed to retrieve students: {str(e)}")
        return []

def match_students_with_canvas(processed_files, canvas_students, matching_mode='name_and_number'):
    """
    Matches processed files with Canvas students using string matching.
    Returns matched and unmatched files.
    """
    matches = []
    unmatched = []
    
    # Create a dictionary of Canvas students for easy lookup
    canvas_student_dict = {student.id: student.name for student in canvas_students}
    
    # Process each file
    for file_info in processed_files:
        if not file_info['success']:
            unmatched.append(file_info)
            continue
            
        # Extract the student name and number from the filename
        filename_parts = file_info['new_filename'].split('_')
        if len(filename_parts) < 2:
            unmatched.append(file_info)
            continue
        
        student_number = filename_parts[0]
        student_name = ' '.join(filename_parts[1:]).replace('.pdf', '')
        
        matched = False
        
        if matching_mode == 'name_and_number':
            # First try to match by NESA number if it's not "UnknownNumber"
            if student_number != "UnknownNumber":
                # Try to find a student with matching NESA number in Canvas
                # This would require the NESA number to be stored in Canvas
                # For now, we'll fall back to name matching
                pass
        
        # If no match by number or if using name_only mode, try name matching
        if not matched:
            best_match = None
            highest_score = 0
            
            # Convert student name to lowercase for comparison
            student_name_lower = student_name.lower()
            
            for canvas_id, canvas_name in canvas_student_dict.items():
                canvas_name_lower = canvas_name.lower()
                
                # Try exact match first
                if student_name_lower == canvas_name_lower:
                    best_match = (canvas_id, canvas_name)
                    highest_score = 100
                    break
                
                # If no exact match, use sequence matcher
                score = difflib.SequenceMatcher(None, student_name_lower, canvas_name_lower).ratio() * 100
                
                if score > highest_score:
                    highest_score = score
                    best_match = (canvas_id, canvas_name)
            
            if best_match and highest_score >= 70:  # Threshold for accepting a match
                canvas_id, canvas_name = best_match
                matches.append({
                    'file_info': file_info,
                    'canvas_student_id': canvas_id,
                    'canvas_student_name': canvas_name,
                    'match_score': highest_score
                })
                matched = True
        
        if not matched:
            unmatched.append(file_info)
    
    return matches, unmatched

def get_pdf_preview(pdf_path):
    """Generate a preview image of the top third of the first page of a PDF"""
    try:
        # Extract top third as PNG
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        rect = page.rect
        top_third = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1 / 3)
        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat, clip=top_third)
        
        # Convert to bytes
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception as e:
        st.error(f"Error generating preview: {str(e)}")
        return None

# Add brand styling
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');
        
        /* Brand Colors */
        :root {
            --white: #FFFFFF;
            --purple: #76309B;
            --violet: #AF47E8;
            --black: #000000;
            --gold: #C9A649;
            --light-purple: #f5f0f7;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Comfortaa', sans-serif !important;
            font-weight: 700 !important;
            color: var(--purple) !important;
        }
        
        p, span, div {
            font-family: 'Comfortaa', sans-serif !important;
            color: var(--black);
        }
        
        .caption {
            font-style: italic;
            color: var(--violet);
        }
        
        /* Buttons */
        .stButton > button {
            font-family: 'Comfortaa', sans-serif !important;
            background-color: var(--purple) !important;
            color: var(--white) !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
            transition: background-color 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background-color: var(--violet) !important;
            color: var(--white) !important;
        }
        
        /* Progress Bar */
        .stProgress > div > div > div {
            background-color: var(--purple) !important;
            transition: width 0.3s ease-in-out !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            font-family: 'Comfortaa', sans-serif !important;
            color: var(--violet) !important;
        }
        
        /* Success/Error messages */
        .success {
            color: var(--purple) !important;
        }
        
        .error {
            color: #FF4B4B !important;
        }
        
        /* Header area with logo */
        .header-container {
            display: flex;
            align-items: center;
            padding: 0.5rem 2rem;
            margin: -4rem -4rem 2rem -4rem;
            background-color: var(--white);
            border-bottom: 2px solid var(--purple);
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .logo {
            height: 35px;
            margin-right: 1rem;
        }
        
        .header-title {
            font-family: 'Montserrat', sans-serif !important;
            color: var(--purple);
            margin: 0;
            padding: 0;
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        /* Main content padding to account for fixed header */
        .main-content {
            margin-top: 5rem;
            padding: 1rem;
        }

        /* Upload status container */
        .upload-status {
            margin: 1rem 0;
            padding: 1.5rem;
            background-color: var(--light-purple);
            border-radius: 8px;
            border: 1px solid var(--purple);
        }

        .upload-status-header {
            font-weight: 600;
            color: var(--purple);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .upload-status-header .status-icon {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background-color: var(--purple);
            display: inline-block;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .upload-complete {
            background-color: #e8f5e9 !important;
            border-color: #4caf50 !important;
        }

        .upload-complete .status-icon {
            background-color: #4caf50 !important;
            animation: none !important;
        }

        /* Input fields */
        .stTextInput > div > div > input {
            font-family: 'Comfortaa', sans-serif !important;
        }

        /* Assignment URL input */
        .url-input input {
            background-color: var(--white);
            border: 2px solid var(--purple);
            border-radius: 4px;
            padding: 0.5rem;
            font-family: 'Comfortaa', sans-serif !important;
            width: 100%;
        }
        
        .url-input input:focus {
            border-color: var(--violet);
            box-shadow: 0 0 0 1px var(--violet);
        }

        /* Hide redundant elements */
        [data-testid="stMarkdownContainer"] h3:first-of-type,
        .main-title {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Digital Marking App",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply brand styling
    local_css()
    
    # Display logo and title in header
    st.markdown("""
        <div class="header-container">
            <img src="data:image/svg+xml;base64,{}" class="logo" alt="AcademIQ Logo">
            <h1 class="header-title">Digital Marking App</h1>
        </div>
        <div class="main-content">
    """.format(base64.b64encode(open('logo.svg', 'rb').read()).decode()), unsafe_allow_html=True)
    
    # Initialize session state
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'timestamp' not in st.session_state:
        st.session_state.timestamp = str(int(time.time()))
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = []
    if 'upload_complete' not in st.session_state:
        st.session_state.upload_complete = False

    # Step 1: File Upload with automatic processing
    if st.session_state.current_step == 1:
        st.markdown('<p class="caption">Select PDF files to process</p>', unsafe_allow_html=True)

        # Create progress tracking container
        progress_container = st.empty()
        with progress_container.container():
            status_class = "upload-status upload-complete" if st.session_state.upload_complete else "upload-status"
            st.markdown(f'<div class="{status_class}">', unsafe_allow_html=True)
            st.markdown("""
                <div class="upload-status-header">
                    <span class="status-icon"></span>
                    <span>Upload Status</span>
                </div>
            """, unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        # File upload section
        uploaded_files = st.file_uploader(
            "Select PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            help="Choose one or more PDF files to process",
            on_change=lambda: setattr(st.session_state, 'upload_complete', False)
        )

        if uploaded_files:
            # Create a unique folder for this session's uploads
            session_folder = os.path.join(UPLOAD_FOLDER, 'splits', st.session_state.timestamp)
            os.makedirs(session_folder, exist_ok=True)

            # Process each uploaded file
            total_files = len(uploaded_files)
            for idx, uploaded_file in enumerate(uploaded_files, 1):
                if allowed_file(uploaded_file.name):
                    # Update status
                    status_text.text(f"Processing: {uploaded_file.name}")
                    
                    # Secure the filename
                    filename = secure_filename(uploaded_file.name)
                    save_path = os.path.join(session_folder, filename)
                    
                    # Save the file
                    if save_uploaded_file(uploaded_file, save_path):
                        if filename not in st.session_state.processed_files:
                            st.session_state.processed_files.append(filename)
                    else:
                        st.error(f"Failed to save {filename}")
                else:
                    st.error(f"Invalid file type: {uploaded_file.name}")
            
            # Mark upload as complete
            st.session_state.upload_complete = True
            progress_bar.progress(1.0)
            status_text.text(f"✅ All {total_files} files processed successfully!")

            # Rest of the code...

if __name__ == "__main__":
    main() 