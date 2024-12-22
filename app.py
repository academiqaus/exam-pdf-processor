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
    unmatched = processed_files.copy()  # Start with all files as unmatched
    
    # Create a dictionary of Canvas students for easy lookup
    canvas_student_dict = {student.id: student.name for student in canvas_students}
    
    # Process each file
    for file_info in processed_files:
        if not file_info.get('success', False):  # Changed from if not file_info['success']
            continue
            
        # Extract the student name and number from the filename
        filename_parts = file_info['new_filename'].split('_')
        if len(filename_parts) < 2:
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
                unmatched.remove(file_info)  # Remove from unmatched list
                matched = True
    
    return matches, unmatched

def get_pdf_preview(pdf_path, page_num=0):
    """Generate a preview image of a specific page in a PDF"""
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if page_num >= total_pages:
            page_num = 0
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
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
                        
                        # Display results
                        st.write("### Processing Results")
                        for result in results:
                            if result['success']:
                                st.success(f"Processed {result['original_filename']} → {result['new_filename']}")
                                with st.expander("Show extracted text"):
                                    st.text(result['extracted_text'])
                            else:
                                st.error(f"Failed to process {result['original_filename']}: {result['error']}")
                        
                        st.session_state.processing_complete = True
                        st.session_state.processing_results = results
                        st.session_state.current_step = 2
                        st.rerun()

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

        # If matching is complete, show only the success message and continue button
        if st.session_state.get('matching_complete', False):
            st.success("All files have been matched successfully!")
            st.markdown('<div style="text-align: center; margin-top: 2rem;">', unsafe_allow_html=True)
            if st.button("Continue to Cover Page Removal", type="primary", key="proceed_to_cover"):
                st.session_state.current_step = 3
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            return

        # Reset matching complete if starting new matching
        if 'matching_started' not in st.session_state:
            st.session_state.matching_complete = False
            st.session_state.matching_started = True
        
        # Canvas API Configuration
        assignment_url = st.text_input(
            "Paste Canvas Assignment URL",
            help="Example: https://canvas.parra.catholic.edu.au/courses/12345/assignments/67890"
        )
        
        # Matching mode selection
        st.markdown('<h4>Select Matching Method</h4>', unsafe_allow_html=True)
        matching_mode = st.radio(
            "Select how you want to match students with their files",
            options=['name_and_number', 'name_only'],
            format_func=lambda x: "Use both Name and NESA Number" if x == 'name_and_number' else "Use Name Only",
            help="'Use both Name and NESA Number': Attempts to match using NESA number first, then falls back to name matching if needed.\n'Use Name Only': Only uses student names for matching."
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
                    # Perform matching
                    matches, unmatched = match_students_with_canvas(
                        st.session_state.processing_results,
                        students,
                        matching_mode
                    )
                    
                    # Display results in a results section
                    st.markdown('<div class="results-section">', unsafe_allow_html=True)
                    st.write("### Matching Results")
                    
                    # Show matches
                    if matches:
                        with st.expander("View Successful Matches", expanded=False):
                            for match in matches:
                                st.success(
                                    f"Matched {match['file_info']['new_filename']} → "
                                    f"{match['canvas_student_name']} (Score: {match['match_score']}%)"
                                )
                    
                    # Handle unmatched files
                    if unmatched:
                        st.write("#### Manual Matching Required")
                        st.write("#### Unmatched Files")
                        
                        # Get list of unmatched students (excluding already matched ones)
                        matched_student_ids = {match['canvas_student_id'] for match in matches}
                        unmatched_students = [
                            student for student in students 
                            if student.id not in matched_student_ids
                        ]
                        
                        # Display unmatched files in a grid
                        cols_per_row = 3
                        unmatched_files = [f for f in unmatched if not f.get('success', False)]
                        
                        for i in range(0, len(unmatched_files), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j, col in enumerate(cols):
                                if i + j < len(unmatched_files):
                                    file_info = unmatched_files[i + j]
                                    with col:
                                        # Show PDF preview
                                        pdf_path = os.path.join(session_folder, file_info['new_filename'])
                                        preview_bytes = get_pdf_preview(pdf_path)
                                        if preview_bytes:
                                            st.image(preview_bytes, use_column_width=True)
                                        
                                        st.write(f"**{file_info['new_filename']}**")
                                        
                                        # Student selection with searchable dropdown
                                        search_options = [{'id': str(s.id), 'name': s.name} for s in unmatched_students]
                                        selected = st.selectbox(
                                            "Match with student",
                                            options=[''] + [str(s.id) for s in unmatched_students],
                                            format_func=lambda x: "Type to search students..." if x == '' else next(
                                                (s['name'] for s in search_options if s['id'] == x),
                                                "Unknown"
                                            ),
                                            key=f"match_{file_info['new_filename']}",
                                            help="Start typing to search for a student"
                                        )
                                        
                                        if selected:
                                            student = next(s for s in unmatched_students if str(s.id) == selected)
                                            matches.append({
                                                'file_info': file_info,
                                                'canvas_student_id': student.id,
                                                'canvas_student_name': student.name,
                                                'match_score': 100  # Manual match
                                            })
                                            unmatched_files.remove(file_info)
                                            st.experimental_rerun()
                        
                        if not unmatched_files:
                            st.success("All files have been matched!")
                            st.session_state.matched_files = matches
                            st.session_state.matches = matches
                            st.session_state.matching_complete = True
                            st.experimental_rerun()
                    else:
                        st.success("All files matched successfully!")
                        st.session_state.matched_files = matches
                        st.session_state.matches = matches
                        st.session_state.matching_complete = True
                        st.experimental_rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)

    # Step 3: Cover Page Removal
    elif st.session_state.current_step == 3:
        # Verify we have matched files
        if not st.session_state.get('matches') and not st.session_state.get('matched_files'):
            st.error("No matched files found. Please complete the matching step first.")
            if st.button("Return to Matching"):
                st.session_state.current_step = 2
                st.session_state.matching_complete = False
                st.experimental_rerun()
            return

        # Use matches if available, otherwise use matched_files
        if not st.session_state.matched_files and st.session_state.get('matches'):
            st.session_state.matched_files = st.session_state.matches

        st.markdown('<div class="caption-container"><p class="caption">Remove Cover Pages<span class="wait-text">Select booklet size to remove cover pages...</span></p></div>', unsafe_allow_html=True)
        
        # Get session folder path
        session_folder = os.path.join(UPLOAD_FOLDER, 'splits', st.session_state.timestamp)
        
        # Create preview folder
        preview_folder = os.path.join(UPLOAD_FOLDER, 'preview', st.session_state.timestamp)
        os.makedirs(preview_folder, exist_ok=True)

        # Booklet size selection
        col1, col2 = st.columns([3, 1])
        with col1:
            booklet_size = st.radio(
                "Select Booklet Size",
                options=['4', '8', '12', 'custom'],
                format_func=lambda x: f"{x} pages" if x != 'custom' else "Custom size",
                horizontal=True,
                help="Select the number of pages in each exam booklet"
            )
        
        with col2:
            if booklet_size == 'custom':
                booklet_size = st.number_input("Enter custom size", min_value=1, value=4, help="Enter the number of pages in each exam booklet")

        if st.button("Process Files", type="primary"):
            try:
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
                    
                    # Show previews in a grid
                    st.markdown("### Preview Processed Files")
                    
                    # Initialize page numbers in session state if not exists
                    for filename in processed_files:
                        if f"current_page_{filename}" not in st.session_state:
                            st.session_state[f"current_page_{filename}"] = 0
                    
                    # Display files in a grid
                    cols_per_row = 3
                    for i in range(0, len(processed_files), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(processed_files):
                                filename = processed_files[i + j]
                                with col:
                                    st.markdown(f"**{filename}**")
                                    
                                    # Get current page number from session state
                                    current_page = st.session_state[f"current_page_{filename}"]
                                    
                                    # Get preview and total pages
                                    pdf_path = os.path.join(preview_folder, filename)
                                    preview_bytes, total_pages = get_pdf_preview(pdf_path, current_page)
                                    
                                    if preview_bytes:
                                        st.image(preview_bytes, use_column_width=True)
                                        
                                        # Page navigation
                                        col1, col2, col3 = st.columns([1, 2, 1])
                                        with col1:
                                            if st.button("◀", key=f"prev_{filename}", help="Previous page"):
                                                new_page = (current_page - 1) if current_page > 0 else (total_pages - 1)
                                                st.session_state[f"current_page_{filename}"] = new_page
                                                st.rerun()
                                        with col2:
                                            st.markdown(f"<div style='text-align: center'>Page {current_page + 1}/{total_pages}</div>", unsafe_allow_html=True)
                                        with col3:
                                            if st.button("▶", key=f"next_{filename}", help="Next page"):
                                                new_page = (current_page + 1) if current_page < total_pages - 1 else 0
                                                st.session_state[f"current_page_{filename}"] = new_page
                                                st.rerun()
                                    
                                    # Individual file download button
                                    with open(pdf_path, "rb") as file:
                                        st.download_button(
                                            label=f"📥 Download",
                                            data=file,
                                            file_name=filename,
                                            mime="application/pdf",
                                            key=f"download_{filename}"
                                        )
                
            except ValueError:
                st.error("Invalid booklet size. Please enter a valid number.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 