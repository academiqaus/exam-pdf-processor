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
import zipfile

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_FOLDER = '/tmp/uploads' if os.getenv('STREAMLIT_SERVER_HEADLESS') == 'true' else 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
DUPLICATE_FOLDER = os.path.join(UPLOAD_FOLDER, 'duplicate_splits')
MODEL = "gpt-4o"  # Updated model name

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
    
    # Ensure duplicate folder exists
    os.makedirs(DUPLICATE_FOLDER, exist_ok=True)
    
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
                base_filename = new_filename.replace('.pdf', '')
                while os.path.exists(new_path) or os.path.exists(duplicate_path):
                    new_filename = f"{base_filename}_{count}.pdf"
                    new_path = os.path.join(session_folder, new_filename)
                    duplicate_path = os.path.join(DUPLICATE_FOLDER, new_filename)
                    count += 1

                try:
                    # First copy to duplicate folder, then rename original
                    shutil.copy2(old_path, duplicate_path)
                    os.rename(old_path, new_path)
                    
                    processed_results.append({
                        'original_filename': filename,
                        'new_filename': new_filename,
                        'extracted_text': extracted_text,
                        'success': True,
                        'student_number': student_number,
                        'student_name': student_name
                    })
                except Exception as e:
                    st.error(f"Error handling file {filename}: {str(e)}")
                    processed_results.append({
                        'original_filename': filename,
                        'error': str(e),
                        'success': False
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
    
    # Store results in session state
    st.session_state.processing_results = processed_results
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

def match_students_with_canvas(processed_files, canvas_students, session_folder, matching_mode='name_and_number'):
    """
    Matches processed files with Canvas students using string matching.
    Returns matched and unmatched files.
    """
    matches = []
    unmatched = []
    
    # Create dictionaries for tracking
    canvas_student_dict = {student.id: student for student in canvas_students}
    processed_canvas_user_ids = set()
    matched_students = set()
    available_canvas_user_ids = set(canvas_student_dict.keys())
    
    # Process each file
    for file_info in processed_files:
        # Skip failed processing results
        if not file_info.get('success', False):
            unmatched.append(file_info)
            continue
            
        # Ensure we're using the AI-processed filename
        if not file_info.get('new_filename'):
            st.error(f"Missing processed filename for {file_info.get('original_filename', 'unknown file')}")
            unmatched.append(file_info)
            continue
            
        # Extract the student name and number from the AI-processed filename
        filename_parts = file_info['new_filename'].split('_')
        if len(filename_parts) < 2:
            unmatched.append(file_info)
            continue
        
        student_number = filename_parts[0]
        student_name = ' '.join(filename_parts[1:]).replace('.pdf', '')
        
        matched = False
        canvas_user_id = None
        
        # Skip files that are already in Canvas User ID format
        if re.match(r'^\d+\.pdf$', file_info['new_filename']):
            canvas_user_id = int(file_info['new_filename'].replace('.pdf', ''))
            if canvas_user_id in canvas_student_dict:
                matched_students.add(canvas_user_id)
                processed_canvas_user_ids.add(canvas_user_id)
                available_canvas_user_ids.discard(canvas_user_id)
                matched = True
                matches.append({
                    'file_info': file_info,
                    'canvas_student_id': canvas_user_id,
                    'canvas_student_name': canvas_student_dict[canvas_user_id].name,
                    'match_score': 100
                })
                continue
        
        # If no match by number or if using name_only mode, try name matching
        if not matched:
            best_match = None
            highest_score = 0
            
            # Convert student name to lowercase for comparison
            student_name_lower = student_name.lower()
            
            # Only try to match with available (unmatched) students
            for canvas_id in available_canvas_user_ids:
                student = canvas_student_dict[canvas_id]
                canvas_name_lower = student.name.lower()
                
                # Try exact match first
                if student_name_lower == canvas_name_lower:
                    best_match = student
                    highest_score = 100
                    break
                
                # If no exact match, use sequence matcher
                score = difflib.SequenceMatcher(None, student_name_lower, canvas_name_lower).ratio() * 100
                
                if score > highest_score:
                    highest_score = score
                    best_match = student
            
            if best_match and highest_score >= 70:  # Threshold for accepting a match
                # Rename the file to use Canvas user ID only
                old_path = os.path.join(session_folder, file_info['new_filename'])
                new_filename = f"{best_match.id}.pdf"  # Simplified filename
                new_path = os.path.join(session_folder, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    file_info['new_filename'] = new_filename  # Update filename in file_info
                    
                    matches.append({
                        'file_info': file_info,
                        'canvas_student_id': best_match.id,
                        'canvas_student_name': best_match.name,
                        'match_score': highest_score
                    })
                    matched = True
                    matched_students.add(best_match.id)
                    processed_canvas_user_ids.add(best_match.id)
                    available_canvas_user_ids.discard(best_match.id)
                except Exception as e:
                    st.error(f"Error renaming file: {str(e)}")
                    unmatched.append(file_info)
        
        if not matched:
            unmatched.append(file_info)
    
    # Store unmatched students in session state
    unmatched_students = {
        student_id: student.name 
        for student_id, student in canvas_student_dict.items() 
        if student_id not in matched_students
    }
    st.session_state.unmatched_students = unmatched_students
    
    return matches, unmatched

@st.cache_data
def cache_pdf_preview(pdf_path, page_num, top_third_only=False):
    """Cached version of PDF preview generation"""
    return get_pdf_preview(pdf_path, page_num, top_third_only)

def get_pdf_preview(pdf_path, page_num=0, top_third_only=False, zoom=4):
    """Generate a preview image of a PDF page"""
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if page_num >= total_pages:
            page_num = 0
        page = doc.load_page(page_num)
        rect = page.rect
        
        # If top_third_only, only show the top third of the page
        if top_third_only:
            rect = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1 / 2)  # Changed to top half for better visibility
            
        mat = fitz.Matrix(zoom, zoom)  # Use the zoom parameter
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)  # Disable alpha for better quality
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes, total_pages
    except Exception as e:
        st.error(f"Error generating preview: {str(e)}")
        return None, 0

def get_cover_pages_to_remove(total_pages, booklet_size):
    """Calculate which cover pages should be removed based on booklet size and total pages"""
    pages_to_remove = []
    
    # Calculate how many booklets are combined
    num_booklets = total_pages // booklet_size
    
    # For each booklet, add its first page (cover page) to the removal list
    for i in range(num_booklets):
        cover_page = i * booklet_size  # 0-based index
        pages_to_remove.append(cover_page)
    
    return sorted(pages_to_remove)

def remove_cover_pages(input_path, output_path, booklet_size):
    """Remove cover pages from a PDF based on booklet size"""
    try:
        # Read PDF
        reader = PyPDF2.PdfReader(input_path)
        writer = PyPDF2.PdfWriter()
        
        # Get total pages and calculate cover pages to remove
        total_pages = len(reader.pages)
        pages_to_remove = get_cover_pages_to_remove(total_pages, booklet_size)
        
        # Add all pages except cover pages
        for i in range(total_pages):
            if i not in pages_to_remove:
                writer.add_page(reader.pages[i])
        
        # Save processed PDF
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
            
    except Exception as e:
        raise Exception(f"Error processing {os.path.basename(input_path)}: {str(e)}")

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
            --light-purple: #f7f0fa;
        }
        
        /* Page background */
        .stApp {
            background: linear-gradient(135deg, var(--light-purple) 0%, var(--white) 100%);
        }
        
        /* Typography */
        .caption-container {
            background: var(--white);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 2rem auto;
            max-width: 800px;
            box-shadow: 0 2px 8px rgba(118,48,155,0.1);
            border: 2px solid var(--purple);
        }
        
        .caption {
            font-family: 'Montserrat', sans-serif !important;
            color: var(--purple) !important;
            font-size: 1.2rem;
            margin: 0;
            text-align: center;
            font-weight: 600;
        }

        .caption .wait-text {
            display: block;
            font-weight: normal;
            font-style: italic;
            font-size: 1rem;
            margin-top: 0.5rem;
            color: var(--violet);
            opacity: 0.9;
        }

        /* URL input styling */
        .url-input {
            background: var(--white);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 2rem auto;
            max-width: 800px;
            box-shadow: 0 2px 8px rgba(118,48,155,0.1);
            border: 2px solid var(--purple);
        }

        .url-input input {
            border: 2px solid var(--purple) !important;
            border-radius: 8px !important;
            padding: 0.6rem 1rem !important;
        }

        .url-input input:focus {
            border-color: var(--violet) !important;
            box-shadow: 0 0 0 1px var(--violet) !important;
        }

        /* Matching section styling */
        .matching-section {
            background: var(--white);
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem auto;
            max-width: 800px;
            box-shadow: 0 2px 8px rgba(118,48,155,0.1);
            border: 2px solid var(--purple);
        }

        .matching-section h4 {
            color: var(--purple);
            margin-bottom: 1rem;
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
        }

        .matching-method {
            margin: 1rem 0;
        }

        .matching-method .stRadio {
            background: transparent;
        }

        .matching-method .stRadio > div {
            gap: 0.5rem;
        }

        .matching-method .stRadio > div > div {
            background: var(--white);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(118,48,155,0.2);
            margin: 0.5rem 0;
        }

        .matching-method .stRadio > div > div:hover {
            border-color: var(--violet);
        }

        .matching-method .stRadio label {
            color: var(--purple) !important;
            font-weight: 500;
        }

        /* Results section */
        .results-section {
            background: var(--white);
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem auto;
            max-width: 800px;
            box-shadow: 0 2px 8px rgba(118,48,155,0.1);
            border: 2px solid var(--purple);
        }

        .results-section h3, .results-section h4 {
            color: var(--purple);
            margin-bottom: 1rem;
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
        }

        /* Header area with logo */
        .header-container {
            display: flex;
            align-items: center;
            padding: 1.5rem 3rem;
            margin: -4rem -4rem 2rem -4rem;
            background: var(--light-purple);
            position: relative;
            z-index: 1;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-bottom: 2px solid var(--purple);
        }
        
        .logo-section {
            position: absolute;
            left: 3rem;
        }
        
        .logo {
            height: 40px;
        }
        
        .title-section {
            flex: 1;
            text-align: center;
        }
        
        .header-title {
            font-family: 'Montserrat', sans-serif !important;
            color: var(--purple) !important;
            margin: 0;
            padding: 0;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        /* Buttons */
        .stButton > button {
            font-family: 'Comfortaa', sans-serif !important;
            background: var(--white) !important;
            color: var(--purple) !important;
            border: 2px solid var(--purple) !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background: var(--light-purple) !important;
            border-color: var(--violet) !important;
            color: var(--violet) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Upload area styling */
        .upload-area {
            background: transparent;
            margin: 2rem 0;
        }

        /* File uploader styling */
        .stFileUploader > div {
            background: var(--white);
            padding: 2rem;
            border-radius: 12px;
            border: 2px solid var(--purple);
            transition: all 0.3s ease;
        }

        .stFileUploader > div:hover {
            border-color: var(--violet);
        }

        /* Hide redundant elements */
        [data-testid="stMarkdownContainer"] h3:first-of-type {
            display: none;
        }

        /* Selectbox and other inputs */
        .stSelectbox > div > div {
            background-color: var(--white);
            border: 2px solid var(--purple);
            border-radius: 8px;
        }

        .stSelectbox > div > div:hover {
            border-color: var(--violet);
        }

        .stFileUploader > div {
            background: var(--white);
            padding: 2rem;
            border-radius: 12px;
            border: 2px dashed var(--purple);
        }

        .stFileUploader > div:hover {
            border-color: var(--violet);
        }
        </style>
    """, unsafe_allow_html=True)

# Add this function for consistent logging
def log_debug(message):
    """Log debug message to both terminal and Streamlit interface"""
    logger.debug(message)
    st.write(f"DEBUG: {message}")

def process_files():
    log_debug("Starting process_files()")
    log_debug(f"Current state: {st.session_state.get('state', 'unknown')}")
    
    if 'state' not in st.session_state:
        st.session_state.state = "initial"
    
    if st.session_state.state == "initial":
        log_debug("In initial state, processing uploaded files")
        process_uploaded_files()
        st.session_state.state = "matching"
        log_debug(f"State changed to: {st.session_state.state}")
        st.rerun()
    
    elif st.session_state.state == "matching":
        log_debug("In matching state, handling manual matches")
        if handle_manual_matches():
            log_debug("Manual matching complete, transitioning to cover page removal")
            st.session_state.state = "cover_page_removal"
            st.rerun()
    
    elif st.session_state.state == "cover_page_removal":
        log_debug("In cover_page_removal state, processing cover pages")
        process_cover_pages()
        st.session_state.state = "complete"
        log_debug(f"State changed to: {st.session_state.state}")
        st.rerun()

def handle_manual_matches():
    log_debug("Starting handle_manual_matches()")
    log_debug(f"Current state: {st.session_state.state}")
    log_debug(f"Current matches: {st.session_state.get('matches', [])}")
    log_debug(f"Current unmatched files: {st.session_state.get('unmatched_files', [])}")
    
    if not st.session_state.get('unmatched_files'):
        log_debug("No unmatched files to process")
        st.session_state.state = "cover_page_removal"
        return True
    
    with st.form("manual_matches"):
        log_debug("Displaying manual matches form")
        st.write("### Manual Matching")
        st.write("Please match the following files with Canvas students:")
        
        matches_made = False
        for i, file in enumerate(st.session_state.unmatched_files):
            st.text_input(f"Match for {file}", key=f"match_{i}")
        
        submitted = st.form_submit_button("Apply Matches and Continue")
        log_debug(f"Form submitted: {submitted}")
        
        if submitted:
            log_debug("Processing manual matches")
            for i, file in enumerate(st.session_state.unmatched_files):
                match = st.session_state.get(f"match_{i}")
                if match:
                    log_debug(f"Adding manual match: {file} -> {match}")
                    if 'matches' not in st.session_state:
                        st.session_state.matches = {}
                    st.session_state.matches[file] = match
                    matches_made = True
            
            if matches_made:
                log_debug("Manual matches processed successfully")
                st.session_state.state = "cover_page_removal"
                return True
            else:
                log_debug("No manual matches were made")
                st.warning("Please make at least one match before continuing.")
    
    return False

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
            <div class="logo-section">
                <img src="data:image/svg+xml;base64,{}" class="logo" alt="AcademIQ Logo">
            </div>
            <div class="title-section">
                <h1 class="header-title">Digital Marking App</h1>
            </div>
        </div>
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
    if 'student_search' not in st.session_state:
        st.session_state.student_search = ""
    if 'upload_progress' not in st.session_state:
        st.session_state.upload_progress = 0
        st.session_state.upload_status = ""
    if 'matched_files' not in st.session_state:
        st.session_state.matched_files = None
    if 'matching_complete' not in st.session_state:
        st.session_state.matching_complete = False
    if 'matches' not in st.session_state:
        st.session_state.matches = None

    # Add session cleanup on browser close/refresh
    if st.session_state.get('cleanup_registered') != True:
        atexit.register(cleanup_old_files)
        st.session_state.cleanup_registered = True

    # Step 1: File Upload with automatic processing
    if st.session_state.current_step == 1:
        st.markdown('''
            <div class="caption-container">
                <p class="caption">
                    Upload Split Student PDF Files
                    <span class="wait-text">Please wait for the "Process Files" button to appear...</span>
                </p>
            </div>
        ''', unsafe_allow_html=True)

        # File upload section
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Select PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            help="Choose one or more PDF files to process"
        )

        if uploaded_files:
            # Create a unique folder for this session's uploads
            session_folder = os.path.join(UPLOAD_FOLDER, 'splits', st.session_state.timestamp)
            os.makedirs(session_folder, exist_ok=True)

            # Process each uploaded file
            all_files_saved = True  # Track if all files are saved successfully
            for uploaded_file in uploaded_files:
                if allowed_file(uploaded_file.name):
                    # Secure the filename
                    filename = secure_filename(uploaded_file.name)
                    save_path = os.path.join(session_folder, filename)
                    
                    # Save the file
                    if save_uploaded_file(uploaded_file, save_path):
                        if filename not in st.session_state.processed_files:
                            st.session_state.processed_files.append(filename)
                    else:
                        st.error(f"Failed to save {filename}")
                        all_files_saved = False
                else:
                    st.error(f"Invalid file type: {uploaded_file.name}")
                    all_files_saved = False

            # Only show the Process Files button if all files are saved successfully
            if all_files_saved and len(st.session_state.processed_files) > 0:
                # Process button
                if st.button("Process Files", type="primary"):
                    with st.spinner("Processing with AcademIQ..."):
                        try:
                            # Use environment variable for API key
                            api_key = st.secrets["OPENAI_API_KEY"]
                            global client
                            client = OpenAI(api_key=api_key)
                            
                            session_folder = os.path.join(UPLOAD_FOLDER, 'splits', st.session_state.timestamp)
                            results = process_files_with_openai(
                                st.session_state.processed_files,
                                session_folder,
                                api_key
                            )
                            
                            # Display results and store in session state
                            st.write("### Processing Results")
                            success_count = 0
                            for result in results:
                                if result['success']:
                                    success_count += 1
                                    st.success(f"Processed {result['original_filename']} → {result['new_filename']}")
                                    with st.expander("Show extracted text"):
                                        st.text(result['extracted_text'])
                                else:
                                    st.error(f"Failed to process {result['original_filename']}: {result['error']}")
                            
                            if success_count > 0:
                                st.session_state.processing_complete = True
                                st.session_state.processing_results = results
                                st.session_state.current_step = 2
                                st.rerun()
                            else:
                                st.error("No files were successfully processed. Please try again.")
                        except Exception as e:
                            st.error(f"An error occurred during processing: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)

    # Step 2: Canvas Student Matching
    elif st.session_state.current_step == 2:
        st.markdown('<div class="caption-container"><p class="caption">Match Processed Files with Canvas Students<span class="wait-text">Please paste the Canvas Assignment URL below to begin matching...</span></p></div>', unsafe_allow_html=True)
        
        # Get session folder path
        session_folder = os.path.join(UPLOAD_FOLDER, 'splits', st.session_state.timestamp)
        
        # Check if we have processing results
        if not st.session_state.processing_results:
            st.error("No processed files found. Please complete the processing step first.")
            if st.button("Return to Upload"):
                st.session_state.current_step = 1
                st.rerun()
            return

        # Canvas API Configuration
        assignment_url = st.text_input(
            "Paste Canvas Assignment URL",
            help="Example: https://canvas.parra.catholic.edu.au/courses/12345/assignments/67890"
        )
        
        if assignment_url:
            # Extract base URL and IDs from assignment URL
            canvas_api_url, course_id, assignment_id = extract_course_assignment_ids(assignment_url)
            if not all([canvas_api_url, course_id, assignment_id]):
                return
                
            # Get Canvas API key from secrets
            canvas_api_key = st.secrets["CANVAS_API_KEY"]
            
            # Initialize Canvas
            canvas = authenticate_canvas(canvas_api_url, canvas_api_key)
            if not canvas:
                return
                
            # Get course students
            students = get_course_students(canvas, course_id)
            if not students:
                return
            
            if st.button("Start Matching", type="primary"):
                with st.spinner("Matching students..."):
                    # Perform matching using stored results
                    matches, unmatched = match_students_with_canvas(
                        st.session_state.processing_results,
                        students,
                        session_folder,
                        'name_only'  # Simplified to just use name matching
                    )
                    
                    # Store matches in session state
                    st.session_state.matches = matches if matches else []
                    st.session_state.unmatched = unmatched if unmatched else []
                    st.session_state.unmatched_students = students
                    st.rerun()

            # Show matching results if we have any
            if st.session_state.get('matches') is not None:
                st.markdown("### Matching Results")
                
                # Show automatic matches
                if st.session_state.matches:
                    with st.expander("View Automatic Matches", expanded=False):
                        for match in st.session_state.matches:
                            # Get the AI-processed name from the file info
                            ai_processed_name = match['file_info'].get('student_name', '').replace('_', ' ')
                            ai_processed_number = match['file_info'].get('student_number', '')
                            ai_output = f"{ai_processed_number}_{ai_processed_name}"
                            st.success(f"Matched {ai_output}.pdf → {match['canvas_student_name']}")

                # Handle manual matching if needed
                if st.session_state.get('unmatched'):
                    st.write("### Manual Matching")
                    st.write(f"{len(st.session_state.unmatched)} files need manual matching")
                    
                    with st.form("manual_matching"):
                        manual_matches = {}
                        
                        for file_info in st.session_state.unmatched:
                            st.write("---")  # Separator between files
                            
                            # Show PDF preview and matching controls side by side
                            col1, col2 = st.columns([1, 1.5])
                            
                            with col1:
                                pdf_path = os.path.join(session_folder, file_info['new_filename'])
                                preview_bytes, _ = get_pdf_preview(pdf_path, page_num=0, top_third_only=True, zoom=2)
                                if preview_bytes:
                                    st.image(preview_bytes, use_column_width=True)
                            
                            with col2:
                                st.write(f"**{file_info['new_filename']}**")
                                if 'student_name' in file_info:
                                    st.write(f"Detected Name: {file_info['student_name'].replace('_', ' ')}")
                                if 'student_number' in file_info:
                                    st.write(f"Detected Number: {file_info['student_number']}")
                                
                                student_options = {f"{s.id}": f"{s.name} (ID: {s.id})" for s in st.session_state.unmatched_students}
                                selected = st.selectbox(
                                    "Select student",
                                    options=[""] + list(student_options.keys()),
                                    format_func=lambda x: "Select student..." if x == "" else student_options.get(x, x),
                                    key=f"match_{file_info['new_filename']}"
                                )
                                if selected:
                                    manual_matches[file_info['new_filename']] = selected
                        
                        # Single form submit button at the end
                        if st.form_submit_button("Apply Manual Matches"):
                            # Process manual matches
                            for filename, student_id in manual_matches.items():
                                file_info = next(f for f in st.session_state.unmatched if f['new_filename'] == filename)
                                student = next(s for s in st.session_state.unmatched_students if str(s.id) == student_id)
                                
                                # Rename file to just Canvas ID
                                old_path = os.path.join(session_folder, filename)
                                new_filename = f"{student.id}.pdf"  # Simplified filename
                                new_path = os.path.join(session_folder, new_filename)
                                
                                try:
                                    os.rename(old_path, new_path)
                                    st.session_state.matches.append({
                                        'file_info': file_info,
                                        'canvas_student_id': student.id,
                                        'canvas_student_name': student.name,
                                        'match_score': 100
                                    })
                                except Exception as e:
                                    st.error(f"Error matching {filename}: {str(e)}")
                            
                            # Update unmatched files
                            matched_filenames = set(manual_matches.keys())
                            st.session_state.unmatched = [
                                f for f in st.session_state.unmatched 
                                if f['new_filename'] not in matched_filenames
                            ]
                            st.rerun()

                # Add continue button that will always work
                if st.button("Continue to Cover Page Removal", type="primary"):
                    st.session_state.current_step = 3
                    st.session_state.matched_files = st.session_state.matches
                    st.rerun()

    # Step 3: Cover Page Removal
    elif st.session_state.current_step == 3:
        st.markdown('<div class="caption-container"><p class="caption">Remove Cover Pages<span class="wait-text">Select booklet size to remove cover pages...</span></p></div>', unsafe_allow_html=True)
        
        # Get session folder path
        session_folder = os.path.join(UPLOAD_FOLDER, 'splits', st.session_state.timestamp)
        
        # Create preview folder
        preview_folder = os.path.join(UPLOAD_FOLDER, 'preview', st.session_state.timestamp)
        os.makedirs(preview_folder, exist_ok=True)

        # Verify we have matched files
        if not st.session_state.get('matches') and not st.session_state.get('matched_files'):
            st.error("No matched files found. Please complete the matching step first.")
            if st.button("Return to Matching"):
                st.session_state.current_step = 2
                st.rerun()
            return

        # Use matches if available, otherwise use matched_files
        if not st.session_state.matched_files and st.session_state.get('matches'):
            st.session_state.matched_files = st.session_state.matches

        # Booklet size selection
        col1, col2 = st.columns([3, 1])
        with col1:
            booklet_size = st.radio(
                "Select Booklet Size",
                options=['no_removal', '4', '8', '12', 'custom'],
                format_func=lambda x: "No cover pages to remove" if x == 'no_removal' else (f"{x} pages" if x != 'custom' else "Custom size"),
                horizontal=True,
                help="Select the number of pages in each exam booklet, or choose 'No cover pages to remove' to skip this step"
            )
        
        with col2:
            if booklet_size == 'custom':
                booklet_size = st.number_input("Enter custom size", min_value=1, value=4, help="Enter the number of pages in each exam booklet")

        if st.button("Process Files", type="primary"):
            try:
                if booklet_size == 'no_removal':
                    # Simply copy files without removing cover pages
                    processed_files = []
                    with st.spinner("Copying files..."):
                        for filename in os.listdir(session_folder):
                            if filename.endswith('.pdf'):
                                input_path = os.path.join(session_folder, filename)
                                output_path = os.path.join(preview_folder, filename)
                                shutil.copy2(input_path, output_path)
                                processed_files.append(filename)
                else:
                    booklet_size = int(booklet_size)
                    processed_files = []
                    
                    with st.spinner("Removing cover pages..."):
                        # Process each PDF
                        for filename in os.listdir(session_folder):
                            if filename.endswith('.pdf'):
                                input_path = os.path.join(session_folder, filename)
                                output_path = os.path.join(preview_folder, filename)
                                
                                try:
                                    remove_cover_pages(input_path, output_path, booklet_size)
                                    processed_files.append(filename)
                                except Exception as e:
                                    st.error(f"Error processing {filename}: {str(e)}")
                
                if processed_files:
                    st.success(f"Successfully processed {len(processed_files)} files!")
                    
                    # Store processed state and files
                    st.session_state.files_processed = True
                    st.session_state.processed_pdfs = processed_files
                    st.session_state.preview_folder = preview_folder
                    
                    # Create a zip file of all processed PDFs
                    zip_path = os.path.join(preview_folder, "processed_exams.zip")
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for filename in processed_files:
                            file_path = os.path.join(preview_folder, filename)
                            zip_file.write(file_path, filename)
                    
                    # Add download button for zip file
                    with open(zip_path, "rb") as zip_file:
                        st.download_button(
                            label="📥 Download All Files (ZIP)",
                            data=zip_file,
                            file_name="processed_exams.zip",
                            mime="application/zip",
                            key="download_all_zip"
                        )
                    
                    # Show summary of removed pages
                    st.markdown("### Summary of Removed Pages")
                    
                    # Create a 3-column grid for students
                    cols_per_row = 3
                    for i in range(0, len(processed_files), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(processed_files):
                                filename = processed_files[i + j]
                                with col:
                                    # Get original PDF page count
                                    original_path = os.path.join(session_folder, filename)
                                    processed_path = os.path.join(preview_folder, filename)
                                    
                                    with fitz.open(original_path) as original_doc:
                                        original_pages = len(original_doc)
                                    with fitz.open(processed_path) as processed_doc:
                                        processed_pages = len(processed_doc)
                                    
                                    # Calculate removed pages
                                    removed_pages = get_cover_pages_to_remove(original_pages, booklet_size)
                                    
                                    # Create student card with summary
                                    st.markdown(f"""
                                        <div style='padding: 1rem; border: 2px solid #76309B; border-radius: 12px; margin-bottom: 1rem; background: white; box-shadow: 0 2px 8px rgba(118,48,155,0.1);'>
                                            <h4 style='color: #76309B; margin: 0 0 0.5rem 0; font-family: Montserrat, sans-serif; font-size: 0.9rem;'>{filename}</h4>
                                            <div style='background: #f7f0fa; padding: 0.5rem; border-radius: 8px; margin-bottom: 0.5rem; font-size: 0.8rem;'>
                                                <p style='margin: 0.25rem 0;'><strong>Pages:</strong> {original_pages} → {processed_pages}</p>
                                                <p style='margin: 0.25rem 0;'><strong>Removed:</strong> {', '.join(str(p + 1) for p in removed_pages)}</p>
                                            </div>
                                            <div style='background: #f7f0fa; padding: 0.5rem; border-radius: 8px; margin: 0.5rem 0;'>
                                                <p style='text-align: center; margin: 0; color: #76309B; font-size: 0.8rem;'>
                                                    Removed Pages Preview
                                                </p>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Process removed pages in pairs using Streamlit columns
                                    sorted_pages = sorted(removed_pages)
                                    for i in range(0, len(sorted_pages), 2):
                                        preview_cols = st.columns(2)
                                        for j, col in enumerate(preview_cols):
                                            if i + j < len(sorted_pages):
                                                page_num = sorted_pages[i + j]
                                                preview_bytes, _ = get_pdf_preview(
                                                    original_path, 
                                                    page_num=page_num,
                                                    top_third_only=False,
                                                    zoom=3
                                                )
                                                if preview_bytes:
                                                    with col:
                                                        st.markdown(f"""
                                                            <div style='border: 1px solid #76309B; border-radius: 4px; padding: 0.25rem; background: white;'>
                                                                <p style='text-align: center; color: #76309B; margin: 0 0 0.25rem 0; font-size: 0.7rem;'>
                                                                    Page {page_num + 1}
                                                                </p>
                                                            </div>
                                                        """, unsafe_allow_html=True)
                                                        st.image(preview_bytes, use_column_width=True)
                                    
                                    st.markdown("""
                                        <div style='margin-top: 0.5rem;'>
                                    """, unsafe_allow_html=True)
                                    
                                    # Add download button
                                    with open(processed_path, "rb") as file:
                                        st.download_button(
                                            label="📥 Download",
                                            data=file,
                                            file_name=filename,
                                            mime="application/pdf",
                                            key=f"download_{filename}",
                                            use_container_width=True
                                        )
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)

            except ValueError:
                st.error("Invalid booklet size. Please enter a valid number.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 