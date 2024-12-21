# Exam PDF Processor

A Streamlit application for processing exam PDFs with OpenAI and Canvas integration.

## Features

- Upload multiple student exam PDFs
- Extract student names and numbers using OpenAI Vision API
- Match students with Canvas course roster
- Manual matching interface with PDF previews
- Automatic file cleanup and organization

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `STREAMLIT_SERVER_HEADLESS`: Set to 'true' for cloud deployment

## Usage

1. Upload student exam PDFs
2. Process with OpenAI to extract student information
3. Match with Canvas students
4. Review and confirm matches