import logging
from pathlib import Path
import pdfplumber
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import gzip
import io
import re
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from tqdm import tqdm
import pytesseract
import gc
import os
import numpy as np
from collections import Counter, defaultdict
import hashlib
import shutil
import cv2
from concurrent.futures import ThreadPoolExecutor
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from getpass import getpass
from openai import OpenAI
from time import sleep
import subprocess  # For Tesseract check

# Set Tesseract-OCR path (configurable via environment variable)
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dental_qa.log')
    ]
)
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    if isinstance(handler, logging.FileHandler):
        handler.setLevel(logging.DEBUG)

# Define paths
output_dir = Path(r"D:\ITO\DENTAL KEYWORDS\keyword2")
pdf_dir = output_dir
text_output = output_dir / "extracted_text.txt"
compressed_text = output_dir / "extracted_text.txt.gz"
faiss_text_index_dir = output_dir / "faiss_text_index"
faiss_image_index_dir = output_dir / "faiss_image_index"
image_dir = output_dir / "extracted_images"
pdf_output_dir = output_dir / "qa_pdfs"
keywords_file = output_dir / "keywords.xlsx"
inverted_index_file = output_dir / "inverted_index.json"
image_chunks_excel = output_dir / "image_chunks.xlsx"

# Ensure directories exist
for directory in [output_dir, pdf_dir, faiss_text_index_dir, faiss_image_index_dir, image_dir, pdf_output_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Initialize cache
output_cache = {}


def preprocess_image_for_ocr(image):
    """Preprocess image to improve OCR accuracy."""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(thresh)


def clean_ocr_text(text, keyword=None):
    """Clean OCR output, remove noise, Q&A patterns, deduplicate sentences, and filter irrelevant terms."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'Q:.*?\n|A:.*?\n|Question:.*?\n|Answer:.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\xFF]', '', text)
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    relevance_keywords = ['sinus', 'maxillary', 'paranasal', 'pneumatic', 'nasal', 'osteomeatal', 'sinusitis', 'teeth',
                          'dental', 'mucous', 'caries', 'infection', 'anatomy', 'pathology', 'mucosa', 'drainage']
    if keyword:
        relevance_keywords.append(keyword.lower())

    deduped_sentences = []
    seen_sentences = set()
    for sentence in sentences:
        sentence_lower = sentence.lower()
        sentence_hash = hashlib.sha256(sentence_lower.encode('utf-8')).hexdigest()
        if sentence_hash not in seen_sentences and any(kw in sentence_lower for kw in relevance_keywords):
            seen_sentences.add(sentence_hash)
            deduped_sentences.append(sentence)

    return " ".join(deduped_sentences).strip()


def has_images(pdf_path):
    """Check if a PDF contains any images using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            image_list = doc[page_num].get_images(full=True)
            if image_list:
                doc.close()
                return True
        doc.close()
        return False
    except Exception as e:
        logger.error(f"Error checking images in {pdf_path}: {str(e)}")
        return False


def sanitize_pdf_name(pdf_name):
    """Sanitize PDF filename for chunk file naming."""
    return re.sub(r'[^\w\s-]', '_', pdf_name).strip()


def extract_text_and_images_from_pdf(pdf_path, image_dir, keyword=None, text_extraction_needed=False):
    """Extract text using pdfplumber and images using PyMuPDF, with PyPDF2 for metadata, only for PDFs with images unless text is needed."""
    logger.info(f"Processing {pdf_path}")
    full_text = []
    image_chunk = {}  # {page_num: [{pdf_name, page_num, path, width, height}]}
    keyword_index = defaultdict(list)
    image_metadata = []  # For FAISS image index
    images_found = False
    seen_sentences = set()

    # Skip PDFs without images unless text extraction is explicitly needed
    if not text_extraction_needed and not has_images(pdf_path):
        logger.info(f"Skipping {pdf_path}: No images found and text extraction not required.")
        return "", {}, {}, []

    try:
        # Check PDF validity and encryption with PyPDF2
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            if pdf_reader.is_encrypted:
                logger.error(f"PDF {pdf_path} is encrypted and cannot be processed.")
                return "", {}, {}, []

        # Extract text with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            for page_num in tqdm(range(num_pages), desc=f"Processing {pdf_path.name}"):
                page_text = []
                page_images = []

                # Text extraction (only if needed or images are present)
                plumber_page = pdf.pages[page_num]
                text = plumber_page.extract_text() or ""
                if text.strip():
                    cleaned_text = clean_ocr_text(text, keyword)
                    if cleaned_text:
                        page_text.append(cleaned_text)
                        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text.strip())
                        for sentence in sentences:
                            words = sentence.lower().split()
                            for word in set(words):
                                if keyword and keyword.lower() in word:
                                    keyword_index[word].append((pdf_path.name, page_num + 1, sentence))

                # Image extraction with PyMuPDF
                doc = fitz.open(pdf_path)
                fitz_page = doc[page_num]
                image_list = fitz_page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    try:
                        base_image = doc.extract_image(xref)
                        if base_image and "image" in base_image:
                            img_width, img_height = base_image["width"], base_image["height"]
                            if img_width < 100 or img_height < 100:
                                logger.debug(f"Skipping low-resolution image {img_index} on page {page_num + 1}")
                                continue
                            img_path = image_dir / f"{sanitize_pdf_name(pdf_path.stem)}_page_{page_num + 1}_img_{img_index}.png"
                            with open(img_path, "wb") as f:
                                f.write(base_image["image"])
                            page_images.append({
                                "pdf_name": pdf_path.name,
                                "page_num": page_num + 1,
                                "path": str(img_path),
                                "width": img_width,
                                "height": img_height
                            })
                            images_found = True
                            logger.info(f"Extracted image {img_path} ({img_width}x{img_height})")
                            # Add image metadata for FAISS
                            image_metadata.append({
                                "text": f"{pdf_path.name} page {page_num + 1} image {img_index}: {cleaned_text[:500] or 'No text'}",
                                "metadata": {
                                    "pdf_name": pdf_path.name,
                                    "page_num": page_num + 1,
                                    "image_path": str(img_path)
                                }
                            })
                    except Exception as e:
                        logger.error(f"Error extracting image {img_index} on page {page_num + 1}: {str(e)}")

                # Skip OCR if images are present or text extraction isn't needed
                if not page_text and not page_images and text_extraction_needed:
                    logger.info(f"Rendering page {page_num + 1} for OCR due to no text and text extraction required")
                    try:
                        pix = fitz_page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # Reduced resolution
                        if pix.width < 100 or pix.height < 100:
                            logger.debug(f"Skipping low-resolution page {page_num + 1} ({pix.width}x{pix.height})")
                            continue
                        img_path = image_dir / f"{sanitize_pdf_name(pdf_path.stem)}_page_{page_num + 1}_full.png"
                        pix.save(img_path)
                        page_images.append({
                            "pdf_name": pdf_path.name,
                            "page_num": page_num + 1,
                            "path": str(img_path),
                            "width": pix.width,
                            "height": pix.height
                        })
                        images_found = True
                        logger.info(f"Saved rendered page image {img_path} ({pix.width}x{pix.height})")
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        img_processed = preprocess_image_for_ocr(img)
                        ocr_text = pytesseract.image_to_string(img_processed, config='--psm 6 --oem 3')
                        cleaned_text = clean_ocr_text(ocr_text, keyword)
                        if cleaned_text:
                            page_text.append(cleaned_text)
                            sentences = re.split(r'(?<=[.!?])\s+', cleaned_text.strip())
                            for sentence in sentences:
                                words = sentence.lower().split()
                                for word in set(words):
                                    if keyword and keyword.lower() in word:
                                        keyword_index[word].append((pdf_path.name, page_num + 1, sentence))
                            # Add OCR text to image metadata
                            image_metadata.append({
                                "text": f"{pdf_path.name} page {page_num + 1} OCR: {cleaned_text[:500] or 'No text'}",
                                "metadata": {
                                    "pdf_name": pdf_path.name,
                                    "page_num": page_num + 1,
                                    "image_path": str(img_path)
                                }
                            })
                        img_processed.close()
                        img.close()
                    except Exception as e:
                        logger.error(f"Error during OCR for page {page_num + 1}: {str(e)}")

                # Combine and deduplicate text
                combined_text = ' '.join(page_text) if page_text else 'No text extracted'
                sentences = re.split(r'(?<=[.!?])\s+', combined_text.strip())
                deduped_sentences = []
                for sentence in sentences:
                    normalized_sentence = re.sub(r'\s+', ' ', sentence.strip())
                    sentence_hash = hashlib.sha256(normalized_sentence.encode('utf-8')).hexdigest()
                    if sentence_hash not in seen_sentences:
                        seen_sentences.add(sentence_hash)
                        deduped_sentences.append(sentence)
                final_text = ' '.join(deduped_sentences) if deduped_sentences else 'No text extracted'
                full_text.append(f"[Page {page_num + 1}] {final_text}")
                if page_images:
                    image_chunk[page_num + 1] = page_images

                doc.close()

        # Save image chunk to JSON
        chunk_file = image_dir / f"{sanitize_pdf_name(pdf_path.stem)}_images.json"
        if image_chunk:
            with open(chunk_file, "w") as f:
                json.dump(image_chunk, f, indent=2)
            logger.info(f"Saved image chunk file: {chunk_file}")
        else:
            logger.warning(f"No images extracted for {pdf_path}. No chunk file created.")

        if not images_found and not text_extraction_needed:
            logger.warning(f"No images found in {pdf_path}. Skipping further processing.")
            return "", {}, {}, []
        elif not images_found:
            logger.info(f"No images found in {pdf_path}, but text extracted due to requirement.")

        return ' '.join(full_text), image_chunk, keyword_index, image_metadata

    except PyPDF2.errors.PdfReadError:
        logger.error(f"Corrupted or invalid PDF: {pdf_path}")
        with open(output_dir / "corrupted_pdfs.txt", "a") as f:
            f.write(f"{pdf_path}\n")
        return "", {}, {}, []
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return "", {}, {}, []


def batch_extract_pdfs(pdf_paths, image_dir, keyword=None, text_extraction_needed=False):
    """Extract text, images, and keyword index from multiple PDFs in parallel, skipping PDFs without images unless text is needed."""
    logger.info(f"Processing {len(pdf_paths)} PDFs")
    all_texts = []
    all_image_chunks = {}
    global_keyword_index = defaultdict(list)
    all_image_metadata = []
    skipped_pdfs = []

    max_workers = min(os.cpu_count(), 4)  # Cap workers for efficiency
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract_text_and_images_from_pdf, pdf_path, image_dir, keyword, text_extraction_needed)
            for pdf_path in pdf_paths]
        for future, pdf_path in tqdm(zip(futures, pdf_paths), total=len(pdf_paths), desc="Extracting PDFs"):
            text, image_chunk, keyword_index, image_metadata = future.result()
            if text or image_chunk:
                all_texts.append(f"[{pdf_path.name}] {text}")
                all_image_chunks[pdf_path.name] = image_chunk
                for kw, entries in keyword_index.items():
                    global_keyword_index[kw].extend(entries)
                all_image_metadata.extend(image_metadata)
            else:
                skipped_pdfs.append(pdf_path.name)

    total_images = sum(sum(len(paths) for paths in chunk.values()) for chunk in all_image_chunks.values())
    logger.info(f"Total images extracted: {total_images}")
    if total_images == 0 and not text_extraction_needed:
        logger.warning("No images extracted from any PDFs. Check PDF content or enable text extraction.")
    if skipped_pdfs:
        logger.warning(f"Skipped {len(skipped_pdfs)} PDFs due to no images or errors: {', '.join(skipped_pdfs)}")

    # Save image chunks to Excel
    chunk_data = []
    for pdf_name, chunks in all_image_chunks.items():
        for page_num, images in chunks.items():
            for img in images:
                chunk_data.append({
                    "PDF_Name": pdf_name,
                    "Page_Number": page_num,
                    "Image_Path": img["path"],
                    "Width": img["width"],
                    "Height": img["height"]
                })
    if chunk_data:
        df = pd.DataFrame(chunk_data)
        df.to_excel(image_chunks_excel, index=False)
        logger.info(f"Saved image chunk data to {image_chunks_excel}")
    else:
        logger.warning(f"No image chunk data to save to {image_chunks_excel}")

    return '\n'.join(all_texts), all_image_chunks, global_keyword_index, all_image_metadata


def check_existing_material():
    """Check if extracted material exists and return available components."""
    existing_components = []
    if text_output.exists():
        existing_components.append("text_extraction")
    if compressed_text.exists():
        existing_components.append("text_compression")
    if faiss_text_index_dir.exists() and (faiss_text_index_dir / "index.faiss").exists():
        existing_components.append("text_index")
    if faiss_image_index_dir.exists() and (faiss_image_index_dir / "index.faiss").exists():
        existing_components.append("image_index")
    if image_dir.exists() and any(image_dir.glob("*.png")):
        existing_components.append("image_extraction")
    if inverted_index_file.exists():
        existing_components.append("inverted_index")
    return existing_components


def prompt_for_rewrite_options(existing_components):
    """Prompt user to select which processes to run during rewrite."""
    print("Existing components found:", ", ".join(existing_components) if existing_components else "None")
    print("Select processes to run during rewrite (enter numbers separated by commas, or 'all' for all processes):")
    processes = [
        "1. Text Extraction",
        "2. Image Extraction",
        "3. Text Vector Store Creation",
        "4. Image Vector Store Creation",
        "5. Inverted Index Creation",
        "6. Keyword Extraction"
    ]
    for proc in processes:
        print(proc)

    while True:
        choice = input("Enter your choices (e.g., '1,3,4' or 'all'): ").strip().lower()
        if choice == 'all':
            return ['text_extraction', 'image_extraction', 'text_index', 'image_index', 'inverted_index',
                    'keyword_extraction']
        try:
            selected = [int(x.strip()) for x in choice.split(',')]
            if not all(1 <= x <= 6 for x in selected):
                raise ValueError
            selected_processes = []
            if 1 in selected:
                selected_processes.append('text_extraction')
            if 2 in selected:
                selected_processes.append('image_extraction')
            if 3 in selected:
                selected_processes.append('text_index')
            if 4 in selected:
                selected_processes.append('image_index')
            if 5 in selected:
                selected_processes.append('inverted_index')
            if 6 in selected:
                selected_processes.append('keyword_extraction')
            return selected_processes
        except ValueError:
            print("Invalid input. Enter numbers between 1 and 6 separated by commas, or 'all'.")


def clear_existing_files(selected_processes):
    """Clear existing files based on selected processes for re-extraction."""
    logger.info(f"Clearing files for selected processes: {', '.join(selected_processes)}")
    backup_dir = output_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    for file in [text_output, compressed_text, inverted_index_file, image_chunks_excel]:
        if file.exists():
            shutil.copy(file, backup_dir / file.name)
    if 'text_index' in selected_processes and faiss_text_index_dir.exists():
        shutil.copytree(faiss_text_index_dir, backup_dir / faiss_text_index_dir.name, dirs_exist_ok=True)
    if 'image_index' in selected_processes and faiss_image_index_dir.exists():
        shutil.copytree(faiss_image_index_dir, backup_dir / faiss_image_index_dir.name, dirs_exist_ok=True)
    if 'image_extraction' in selected_processes and image_dir.exists():
        shutil.copytree(image_dir, backup_dir / image_dir.name, dirs_exist_ok=True)

    if 'text_extraction' in selected_processes:
        if text_output.exists():
            text_output.unlink()
        if compressed_text.exists():
            compressed_text.unlink()
    if 'text_index' in selected_processes and faiss_text_index_dir.exists():
        shutil.rmtree(faiss_text_index_dir)
    if 'image_index' in selected_processes and faiss_image_index_dir.exists():
        shutil.rmtree(faiss_image_index_dir)
    if 'image_extraction' in selected_processes and image_dir.exists():
        shutil.rmtree(image_dir)
    if 'inverted_index' in selected_processes and inverted_index_file.exists():
        inverted_index_file.unlink()
    if 'image_extraction' in selected_processes and image_chunks_excel.exists():
        image_chunks_excel.unlink()
    for directory in [faiss_text_index_dir, faiss_image_index_dir, image_dir]:
        directory.mkdir(parents=True, exist_ok=True)


def sanitize_filename(question):
    """Sanitize question for use as a filename."""
    question = re.sub(r'[^\w\s-]', '', question.replace(' ', '_')).strip()
    return question[:50]


def setup_vector_stores(extracted_text, image_metadata, faiss_text_index_dir, faiss_image_index_dir,
                        run_text_index=True, run_image_index=True):
    """Set up FAISS vector stores for text and images."""
    logger.info("Setting up FAISS vector stores")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store_text = None
    vector_store_image = None

    # Text vector store
    if run_text_index and extracted_text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=50)
        text_chunks = text_splitter.split_text(extracted_text)
        text_metadatas = []
        for i, chunk in enumerate(text_chunks):
            source = 'unknown'
            if "[" in chunk:
                try:
                    source = chunk.split("[")[1].split("]")[0].strip()
                    # Validate source as a valid PDF name
                    if not source.endswith('.pdf') or any(c in source for c in '[]{}'):
                        source = 'unknown'
                except IndexError:
                    source = 'unknown'
            text_metadatas.append({"chunk_id": i, "source": source})
        vector_store_text = FAISS.from_texts(text_chunks, embeddings, metadatas=text_metadatas)
        vector_store_text.save_local(faiss_text_index_dir)
        logger.info(f"Text FAISS index saved to {faiss_text_index_dir}")

    # Image vector store
    if run_image_index and image_metadata:
        image_texts = [item["text"] for item in image_metadata]
        image_metadatas = [item["metadata"] for item in image_metadata]
        vector_store_image = FAISS.from_texts(image_texts, embeddings, metadatas=image_metadatas)
        vector_store_image.save_local(faiss_image_index_dir)
        logger.info(f"Image FAISS index saved to {faiss_image_index_dir}")

    return vector_store_text, vector_store_image


def setup_qa_chain():
    """Set up the QA chain with OpenAI client for NVIDIA Inference API."""
    logger.info("Setting up QA chain with OpenAI client for NVIDIA Inference API")
    model_name = "mistralai/mixtral-8x7b-instruct-v0.1"

    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    if not NVIDIA_API_KEY:
        NVIDIA_API_KEY = getpass("Enter your NVIDIA API key: ").strip()
        if not NVIDIA_API_KEY:
            logger.error("No API key provided.")
            raise ValueError("An NVIDIA API key is required for Inference API.")

    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}")
        raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

    try:
        logger.debug("Validating NVIDIA API key with test request...")
        test_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=50,
            stream=True
        )
        full_response = ""
        for chunk in test_response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        if not full_response:
            raise ValueError("Empty response from NVIDIA API.")
        logger.info("NVIDIA API connection successful")
    except Exception as e:
        logger.error(f"Error validating NVIDIA API: {str(e)}")
        raise ValueError(
            f"Failed to set up QA chain: {str(e)}. Check your NVIDIA API key."
        )

    template = """
    [INST] You are a dental expert. Based on the following context from PDFs and image metadata, provide a structured answer with a summary (1-2 sentences) followed by detailed points covering all relevant information about the keyword. Text Context: {text_context} Image Context: {image_context} Question: {question} [/INST]
    """
    prompt = PromptTemplate(template=template, input_variables=["text_context", "image_context", "question"])
    return client, prompt


def generate_pdf_answer(question, answer, keyword_matches, image_paths, output_dir, styles):
    """Generate a PDF with the question, answer, keyword matches, and relevant images."""
    sanitized_question = sanitize_filename(question)
    pdf_path = output_dir / f"{sanitized_question}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    story = []

    story.append(Paragraph(f"Question: {question}", styles['Heading1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Answer:", styles['Heading2']))
    story.append(Paragraph(answer, styles['BodyText']))
    story.append(Spacer(1, 24))

    story.append(Paragraph("Keyword Matches:", styles['Heading2']))
    for pdf_name, page_num, sentence in keyword_matches[:50]:
        story.append(Paragraph(f"PDF: {pdf_name}, Page: {page_num}", styles['BodyText']))
        story.append(Paragraph(f"Text: {sentence}", styles['BodyText']))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Relevant Images:", styles['Heading2']))
    if not image_paths:
        story.append(Paragraph("No relevant images found.", styles['BodyText']))
        logger.warning(f"No images included in PDF for question: {question}")
    else:
        max_width = 400
        max_height = 600
        for img_path in image_paths[:5]:
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                    if img_width > max_width or img_height > max_height:
                        scale = min(max_width / img_width, max_height / img_height)
                        new_width = img_width * scale
                        new_height = img_height * scale
                    else:
                        new_width = img_width
                        new_height = img_height
                    img_rl = ReportLabImage(img_path, width=new_width, height=new_height)
                    story.append(img_rl)
                    story.append(Paragraph(f"Image from {img_path}", styles['BodyText']))
                    story.append(Spacer(1, 12))
            except Exception as e:
                logger.error(f"Error adding image {img_path} to PDF: {str(e)}")
                story.append(Paragraph(f"Failed to include image: {img_path} (Error: {str(e)})", styles['BodyText']))
                story.append(Spacer(1, 12))

    try:
        doc.build(story)
        logger.info(f"Generated PDF: {pdf_path}")
    except Exception as e:
        logger.error(f"Error building PDF {pdf_path}: {str(e)}")
        raise
    return pdf_path


def extract_keywords(text, top_n=10):
    """Extract top keywords using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [keyword for keyword, score in keyword_scores]


def score_image_relevance(pdf_name, page_num, keyword, inverted_index):
    """Score image relevance based on keyword matches in inverted index."""
    score = 0
    keyword_lower = keyword.lower()

    for kw in inverted_index:
        if keyword_lower in kw.lower() or kw.lower() in keyword_lower:
            for match_pdf, match_page, _ in inverted_index[kw]:
                if match_pdf == pdf_name and match_page == page_num:
                    score += 1
                    logger.debug(f"Boosted score for page match: {kw} in {pdf_name}, page {page_num}")
                    break

    if keyword_lower in pdf_name.lower():
        score += 0.5
        logger.debug(f"Boosted score for PDF name match: {keyword_lower} in {pdf_name}")

    return score


def load_image_chunk(pdf_name, image_dir):
    """Load image chunk file for a given PDF."""
    if not pdf_name or not pdf_name.endswith('.pdf') or any(c in pdf_name for c in '[]{}'):
        logger.debug(f"Skipping invalid PDF name for chunk lookup: {pdf_name}")
        return {}
    chunk_file = image_dir / f"{sanitize_pdf_name(Path(pdf_name).stem)}_images.json"
    if chunk_file.exists():
        with open(chunk_file, "r") as f:
            return json.load(f)
    logger.warning(f"No image chunk file found for {pdf_name}")
    return {}


def interactive_qa(vector_store_text, vector_store_image, image_chunks, inverted_index, llm, prompt, output_dir):
    """Interactive QA loop using both text and image FAISS indices."""
    styles = getSampleStyleSheet()
    if not image_chunks:
        logger.warning("No image chunks available. Image retrieval will be limited.")
    if not vector_store_image:
        logger.warning("No image FAISS index available. Image retrieval will rely on chunks only.")

    invalid_sources = set()

    while True:
        question = input("Enter your question or keyword (or 'exit' to quit): ").strip()
        if question.lower() == 'exit':
            if invalid_sources:
                logger.warning(
                    f"Encountered {len(invalid_sources)} invalid sources during QA: {', '.join(sorted(invalid_sources))}")
            break

        cache_key = hashlib.sha256(question.encode('utf-8')).hexdigest()
        if cache_key in output_cache:
            answer, keyword_matches, image_paths = output_cache[cache_key]
            print(f"Cached Answer: {answer}")
            print("Keyword Matches:")
            for pdf_name, page_num, sentence in keyword_matches[:10]:
                print(f"PDF: {pdf_name}, Page: {page_num}, Text: {sentence}")
            for img_path in image_paths:
                print(f"Image: {img_path}")
            pdf_path = generate_pdf_answer(question, answer, keyword_matches, image_paths, output_dir, styles)
            print(f"PDF generated: {pdf_path}")
            continue

        # Initialize variables
        keyword = question.split()[-1].lower() if question else None
        if not keyword:
            print("Please provide a valid question or keyword.")
            continue

        keyword_matches = []
        for kw in inverted_index:
            if keyword in kw.lower():
                keyword_matches.extend(inverted_index[kw])

        # Text search
        text_context = ""
        docs_with_scores = []
        if vector_store_text:
            docs_with_scores = vector_store_text.similarity_search_with_score(question, k=100)
            text_context = "\n".join([doc.page_content[:500] for doc, _ in docs_with_scores[:10]])
            # Generate Excel for text search results
            text_results = []
            for doc, score in docs_with_scores:
                chunk_id = doc.metadata.get('chunk_id', 'unknown')
                source = doc.metadata.get('source', 'unknown')
                page_match = re.search(r'\[Page (\d+)\]', doc.page_content)
                page_number = int(page_match.group(1)) if page_match else None
                image_paths_for_chunk = []
                if source != 'unknown' and source.endswith('.pdf'):
                    chunk = load_image_chunk(source, image_dir)
                    if page_number and str(page_number) in chunk:
                        image_paths_for_chunk = [img["path"] for img in chunk[str(page_number)]]
                else:
                    invalid_sources.add(source)
                text_results.append({
                    'Query': question,
                    'Chunk_ID': chunk_id,
                    'Source': source,
                    'Page_Number': page_number if page_number else 'unknown',
                    'Text': doc.page_content[:1000],
                    'Similarity_Score': score,
                    'Image_Paths': '; '.join(image_paths_for_chunk)
                })
            text_df = pd.DataFrame(text_results[:1000])
            excel_path = output_dir / f"text_search_{sanitize_filename(question)}.xlsx"
            text_df.to_excel(excel_path, index=False)
            logger.info(f"Text search results saved to {excel_path}")
        else:
            # Image search
            logger.info("No text FAISS index available, relying on image search and inverted index.")

        image_context = ""
        image_paths = []
        if vector_store_image:
            image_docs_with_scores = vector_store_image.similarity_search_with_score(question, k=50)
            image_context = "\n".join([doc.page_content[:500] for doc, _ in image_docs_with_scores[:5]])
            image_paths = [doc.metadata["image_path"] for doc, _ in image_docs_with_scores[:5]]
            # Combine with chunk-based image retrieval
            for doc, _ in docs_with_scores:
                source = doc.metadata.get('source', 'unknown')
                if source == 'unknown' or not source.endswith('.pdf'):
                    invalid_sources.add(source)
                    continue
                page_match = re.search(r'\[Page (\d+)\]', doc.page_content)
                page_number = int(page_match.group(1)) if page_match else None
                chunk = load_image_chunk(source, image_dir)
                if page_number and str(page_number) in chunk:
                    image_paths.extend([img["path"] for img in chunk[str(page_number)]])
            image_paths = list(dict.fromkeys(image_paths))[:5]
            # Score images
            scored_images = [(doc.metadata["image_path"], score_image_relevance(
                doc.metadata["pdf_name"],
                doc.metadata["page_num"],
                keyword,
                inverted_index
            )) for doc, _ in image_docs_with_scores
                             if
                             "pdf_name" in doc.metadata and "page_num" in doc.metadata and "image_path" in doc.metadata]
            scored_images.sort(key=lambda x: x[1], reverse=True)
            image_paths = [path for path, _ in scored_images][:5]
        else:
            # Fallback to chunk-based retrieval
            for pdf_name, page_num, _ in keyword_matches:
                if not pdf_name.endswith('.pdf'):
                    invalid_sources.add(pdf_name)
                    continue
                chunk = load_image_chunk(pdf_name, image_dir)
                if str(page_num) in chunk:
                    image_paths.extend([img["path"] for img in chunk[str(page_num)]])
            image_paths = list(dict.fromkeys(image_paths))[:5]
            scored_images = [(path, score_image_relevance(
                pdf_name,
                page_num,
                keyword,
                inverted_index
            )) for pdf_name, chunk in image_chunks.items()
                             for page_num, images in chunk.items()
                             for img in images if img["path"] in image_paths]
            scored_images.sort(key=lambda x: x[1], reverse=True)
            image_paths = [path for path, _ in scored_images][:5]

        if not image_paths:
            logger.warning(f"No relevant images found for query: {question}")

        # Handle keyword extraction
        if "extract" in question.lower() and "keywords" in question.lower():
            logger.info("Processing keyword extraction request")
            keywords = extract_keywords(text_context or image_context, top_n=20)
            dental_keywords = ['dental', 'tooth', 'teeth', 'pulp', 'dentin', 'enamel', 'caries', 'sinus', 'maxillary',
                               'sinusitis', 'mucosa', 'infection', 'anatomy', 'pathology', 'odont', 'oral']
            relevant_keywords = [kw for kw in keywords if any(dkw in kw.lower() for dkw in dental_keywords)]
            if not relevant_keywords:
                relevant_keywords = keywords[:10]

            keyword_matches = []
            for kw in relevant_keywords:
                for inv_kw in inverted_index:
                    if kw.lower() in inv_kw.lower():
                        keyword_matches.extend(inverted_index[inv_kw])

            summary = "Summary: The following dental-related keywords were extracted from the PDFs and images."
            details = ["Detailed Points:"]
            for kw in relevant_keywords:
                kw_context = text_context or image_context
                details.append(f"- {kw}: {kw_context[:200]}..." if len(kw_context) > 200 else f"- {kw}: {kw_context}")
            answer = "\n".join([summary, ""] + details)
            if image_paths:
                answer += "\n\nRelevant Image Paths:\n" + "\n".join([f"- {path}" for path in image_paths])
        else:
            # Generate answer using LLM
            try:
                input_dict = {
                    "text_context": text_context + "\nKeyword Matches: " + "; ".join(
                        [f"{pdf}: {page}: {text}" for pdf, page, text in keyword_matches[:10]]),
                    "image_context": image_context or "No image context available",
                    "question": question
                }
                formatted_prompt = prompt.format(**input_dict)
                logger.debug("Sending request to NVIDIA API via OpenAI client")
                for attempt in range(3):
                    try:
                        response = llm.chat.completions.create(
                            model="mistralai/mixtral-8x7b-instruct-v0.1",
                            messages=[{"role": "user", "content": formatted_prompt}],
                            max_tokens=1024,
                            temperature=0.5,
                            top_p=1,
                            stream=True
                        )
                        raw_answer = ""
                        for chunk in response:
                            if chunk.choices[0].delta.content is not None:
                                raw_answer += chunk.choices[0].delta.content
                        if not raw_answer:
                            logger.error("Empty response from NVIDIA API.")
                            raise ValueError("Empty response from NVIDIA API.")
                        break
                    except Exception as e:
                        if "429" in str(e):
                            logger.warning(f"Rate limit hit, retrying in {2 ** attempt} seconds...")
                            sleep(2 ** attempt)
                            continue
                        raise
                if "[/INST]" in raw_answer:
                    answer = raw_answer.split("[/INST]")[-1].strip()
                else:
                    answer = raw_answer.strip()
                if not answer:
                    answer = "No relevant information found."
                if not image_paths:
                    answer += "\n\nWarning: No relevant images found."
            except Exception as e:
                logger.error(f"Error generating answer: {str(e)}")
                answer = f"Error generating answer: {str(e)}"

        keyword_df = pd.DataFrame(keyword_matches, columns=["PDF", "Page", "Text"])
        keyword_df["Image Paths"] = ""
        for i, (pdf_name, page_num, _) in enumerate(keyword_matches):
            if not pdf_name.endswith('.pdf'):
                invalid_sources.add(pdf_name)
                continue
            chunk = load_image_chunk(pdf_name, image_dir)
            matching_images = [img["path"] for img in chunk.get(str(page_num), [])]
            keyword_df.at[i, "Image Paths"] = "; ".join(matching_images)
        keyword_df.to_excel(keywords_file, index=False)
        logger.info(f"Keyword matches saved to {keywords_file}")

        output_cache[cache_key] = (answer, keyword_matches, image_paths)
        print(f"Answer: {answer}")
        print("Keyword Matches (sample):")
        for pdf_name, page_num, sentence in keyword_matches[:10]:
            print(f"PDF: {pdf_name}, Page: {page_num}, Text: {sentence}")
        for img_path in image_paths:
            print(f"Image: {img_path}")
        if not image_paths:
            print("No relevant images found.")

        pdf_path = generate_pdf_answer(question, answer, keyword_matches, image_paths, output_dir, styles)
        print(f"PDF generated: {pdf_path}")
# Add this to PYTHONN1.py (or pythonn1.py) before the main() function

def main():
    """Main function to process PDFs and start QA with selective process execution."""
    logger.info("Starting AEC Dental QA System")

    # Verify Tesseract installation
    try:
        subprocess.run([pytesseract.pytesseract.tesseract_cmd, "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Tesseract-OCR not found. Please install or update TESSERACT_PATH.")
        exit(1)

    existing_components = check_existing_material()
    if existing_components:
        print(f"Existing components found: {', '.join(existing_components)}")
        while True:
            choice = input(
                "Do you want to rewrite (re-extract data) or continue with existing material? (Type 'rewrite' or 'continue'): ").strip().lower()
            if choice in ['rewrite', 'continue']:
                break
            print("Invalid input. Please type 'rewrite' or 'continue'.")
    else:
        choice = 'rewrite'

    if choice == 'rewrite':
        print("Choose processes to re-run. Existing data for selected processes will be overwritten.")
        print("Use 'all' to rerun everything, or select specific processes (e.g., '1,3').")
        selected_processes = prompt_for_rewrite_options(existing_components)
        clear_existing_files(selected_processes)

        pdf_paths = list(pdf_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.error(f"No PDFs found in {pdf_dir}")
            return

        extracted_text = ""
        image_chunks = {}
        global_keyword_index = defaultdict(list)
        image_metadata = []

        if any(p in selected_processes for p in ['text_extraction', 'image_extraction', 'inverted_index']):
            text_needed = 'text_extraction' in selected_processes or 'text_index' in selected_processes or 'keyword_extraction' in selected_processes
            extracted_text, image_chunks, global_keyword_index, image_metadata = batch_extract_pdfs(
                pdf_paths, image_dir, text_extraction_needed=text_needed
            )
            if text_needed:
                with open(text_output, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                with open(text_output, "rb") as f_in:
                    with gzip.open(compressed_text, "wb") as f_out:
                        f_out.writelines(f_in)
            if 'inverted_index' in selected_processes:
                with open(inverted_index_file, "w") as f:
                    json.dump(global_keyword_index, f)
                logger.info(f"Inverted index saved to {inverted_index_file}")

        if 'keyword_extraction' in selected_processes:
            if not extracted_text:
                if text_output.exists():
                    with open(text_output, "r", encoding="utf-8") as f:
                        extracted_text = f.read()
                else:
                    logger.error("Text extraction required for keyword extraction but not selected or available.")
                    return
            keywords = extract_keywords(extracted_text)
            pd.DataFrame(keywords, columns=["Keyword"]).to_excel(keywords_file, index=False)
            logger.info(f"Keywords saved to {keywords_file}")

        vector_store_text, vector_store_image = setup_vector_stores(
            extracted_text, image_metadata, faiss_text_index_dir, faiss_image_index_dir,
            run_text_index='text_index' in selected_processes,
            run_image_index='image_index' in selected_processes
        )
    else:
        if not text_output.exists():
            logger.error(f"Text output file {text_output} does not exist. Please choose 'rewrite'.")
            return
        with open(text_output, "r", encoding="utf-8") as f:
            extracted_text = f.read()

        if not inverted_index_file.exists():
            logger.error(f"Inverted index file {inverted_index_file} does not exist. Please choose 'rewrite'.")
            return
        with open(inverted_index_file, "r") as f:
            global_keyword_index = json.load(f)

        image_chunks = {}
        image_metadata = []
        if image_dir.exists() and any(image_dir.glob("*_images.json")):
            for chunk_file in image_dir.glob("*_images.json"):
                pdf_name = chunk_file.stem.replace("_images", "") + ".pdf"
                with open(chunk_file, "r") as f:
                    chunk = json.load(f)
                    image_chunks[pdf_name] = chunk
                    for page_num, images in chunk.items():
                        for img in images:
                            # Use filename and page as proxy text for embedding
                            proxy_text = f"{pdf_name} page {page_num} image: {img['path']}"
                            image_metadata.append({
                                "text": proxy_text,
                                "metadata": {
                                    "pdf_name": pdf_name,
                                    "page_num": int(page_num),
                                    "image_path": img["path"]
                                }
                            })
            total_images = sum(sum(len(paths) for paths in chunk.values()) for chunk in image_chunks.values())
            logger.info(f"Loaded {total_images} images from {len(image_chunks)} chunk files in {image_dir}")
        else:
            logger.warning(f"No image chunk files found in {image_dir}. Image retrieval will be limited.")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store_text = None
        vector_store_image = None
        if (faiss_text_index_dir / "index.faiss").exists():
            vector_store_text = FAISS.load_local(faiss_text_index_dir, embeddings, allow_dangerous_deserialization=True)
        else:
            logger.warning(f"Text FAISS index {faiss_text_index_dir / 'index.faiss'} does not exist.")
        if (faiss_image_index_dir / "index.faiss").exists():
            vector_store_image = FAISS.load_local(faiss_image_index_dir, embeddings,
                                                  allow_dangerous_deserialization=True)
        else:
            logger.warning(f"Image FAISS index {faiss_image_index_dir / 'index.faiss'} does not exist.")

    llm, prompt = setup_qa_chain()
    interactive_qa(vector_store_text, vector_store_image, image_chunks, global_keyword_index, llm, prompt,
                   pdf_output_dir)


if __name__ == "__main__":
    main()