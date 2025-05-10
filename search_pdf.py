import os
import sys
import re
import subprocess
import json
import pytesseract
from datetime import datetime
import logging
from pathlib import Path
import pandas as pd
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import PYTHONN1
import PYTHONN2

# Configure pytesseract path
pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('search_pdf.log')
    ]
)
logger = logging.getLogger(__name__)

# Define paths from PYTHONN1.py and PYTHONN2.py
output_dir = Path(r"D:\ITO\DENTAL KEYWORDS\keyword2")
pdf_output_dir = output_dir / "qa_pdfs"
excel_file_path = r"D:\ITO\DENTAL KEYWORDS\keyword2\Output_With_Image_Links.xlsx"
image_dir = output_dir / "extracted_images"
text_output = output_dir / "extracted_text.txt"
inverted_index_file = output_dir / "inverted_index.json"
faiss_text_index_dir = output_dir / "faiss_text_index"
faiss_image_index_dir = output_dir / "faiss_image_index"

# Ensure output directory exists
pdf_output_dir.mkdir(parents=True, exist_ok=True)

# Global resources to avoid reloading
embeddings = None
vector_store_text = None
vector_store_image = None
llm = None
prompt = None

def sanitize_filename(keyword):
    keyword = keyword.lower()
    keyword = re.sub(r'[^a-z0-9]', '_', keyword)
    keyword = re.sub(r'_+', '_', keyword)
    return keyword.strip('_')

def create_header(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 10)
    canvas.drawString(30, doc.pagesize[1] - 30, f"Unified Search Report for Keyword: {doc.keyword}")
    canvas.drawString(doc.pagesize[0] - 100, doc.pagesize[1] - 30, f"Page {doc.page}")
    canvas.restoreState()

def create_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    canvas.drawString(30, 30, f"Generated on: {timestamp}")
    canvas.restoreState()

def generate_unified_pdf(qa_answer, qa_keyword_matches, qa_image_paths, products, keyword, output_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_name = f"unified_{sanitize_filename(keyword)}_{timestamp}"
    output_path = output_dir / f"{pdf_name}.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter, topMargin=50, bottomMargin=50)
    doc.keyword = keyword
    story = []

    styles = getSampleStyleSheet()
    title_style = styles['Title']
    normal_style = styles['Normal']
    heading_style = ParagraphStyle(
        name='Heading',
        parent=normal_style,
        fontSize=12,
        spaceAfter=12,
        alignment=TA_CENTER
    )

    # Cover Page
    story.append(Paragraph(f"Unified Search Report", title_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(f"Keyword: {keyword}", heading_style))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", heading_style))
    story.append(Spacer(1, 1 * inch))

    # Section 1: QA Results from PDFs
    story.append(Paragraph("Dental QA Results", heading_style))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Answer:", normal_style))
    story.append(Paragraph(qa_answer or "No answer generated.", normal_style))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Keyword Matches:", normal_style))
    if qa_keyword_matches:
        for pdf_name, page_num, sentence in qa_keyword_matches[:50]:
            story.append(Paragraph(f"PDF: {pdf_name}, Page: {page_num}", normal_style))
            story.append(Paragraph(f"Text: {sentence}", normal_style))
            story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No keyword matches found.", normal_style))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Relevant Images:", normal_style))
    failed_images = []
    if qa_image_paths:
        max_width = 400
        max_height = 600
        for img_path in qa_image_paths[:5]:
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                    scale = min(max_width / img_width, max_height / img_height)
                    new_width = img_width * scale
                    new_height = img_height * scale
                    img_rl = ReportLabImage(img_path, width=new_width, height=new_height)
                    story.append(img_rl)
                    story.append(Paragraph(f"Image from {img_path}", normal_style))
                    story.append(Spacer(1, 12))
            except Exception as e:
                failed_images.append((img_path, str(e)))
                story.append(Paragraph(f"Failed to include image: {img_path} (Error: {str(e)})", normal_style))
                story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No relevant images found.", normal_style))
        logger.warning(f"No images included in PDF for keyword: {keyword}")
    story.append(Spacer(1, 0.5 * inch))

    # Section 2: Product Search Results
    story.append(Paragraph("Product Search Results", heading_style))
    story.append(Spacer(1, 0.25 * inch))
    if not products.empty:
        for index, row in products.iterrows():
            name = str(row['Name']) if not pd.isna(row['Name']) else "N/A"
            category = str(row['Category']) if not pd.isna(row['Category']) else "N/A"
            description = str(row['Description']) if not pd.isna(row['Description']) else "N/A"
            price = str(row['Price']) if not pd.isna(row['Price']) else "N/A"
            link = str(row['Link']) if not pd.isna(row['Link']) else "N/A"
            features = str(row['Features']) if not pd.isna(row['Features']) else "N/A"

            data = [
                ["Name:", Paragraph(name, normal_style)],
                ["Category:", Paragraph(category, normal_style)],
                ["Description:", Paragraph(description, normal_style)],
                ["Price:", Paragraph(price, normal_style)],
                ["Link:", Paragraph(link, normal_style)],
                ["Features:", Paragraph(features, normal_style)],
            ]
            table = Table(data, colWidths=[1.5 * inch, 5.5 * inch])
            table.setStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ])
            story.append(table)
            story.append(Spacer(1, 0.25 * inch))

            image_columns = [f'Image Path {i}' for i in range(1, 11)]
            valid_images = []
            for col in image_columns:
                if col in row and not pd.isna(row[col]):
                    path = str(row[col]).strip()
                    if os.path.isfile(path) and path.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            img = ReportLabImage(path)
                            img.drawHeight = 1.5 * inch
                            img.drawWidth = 1.5 * inch
                            valid_images.append(img)
                        except Exception as e:
                            failed_images.append((path, str(e)))
                    else:
                        failed_images.append((path, "File not found or invalid format"))

            if valid_images:
                image_rows = [valid_images[i:i + 3] for i in range(0, len(valid_images), 3)]
                for row_images in image_rows:
                    image_data = [[img for img in row_images]]
                    image_table = Table(image_data, colWidths=[1.75 * inch] * len(row_images))
                    image_table.setStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ])
                    story.append(image_table)
                    story.append(Spacer(1, 0.25 * inch))
            story.append(Spacer(1, 0.5 * inch))
    else:
        story.append(Paragraph("No products found.", normal_style))
        story.append(Spacer(1, 0.25 * inch))

    # Failed Images
    if failed_images:
        story.append(Paragraph("Image Inclusion Status", heading_style))
        story.append(Spacer(1, 0.25 * inch))
        data = [["Image Path", "Reason for Failure"]] + [[path, reason] for path, reason in failed_images]
        table = Table(data, colWidths=[4 * inch, 3 * inch])
        table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ])
        story.append(table)

    doc.build(story, onFirstPage=lambda canvas, doc: (create_header(canvas, doc), create_footer(canvas, doc)),
              onLaterPages=lambda canvas, doc: (create_header(canvas, doc), create_footer(canvas, doc)))
    logger.info(f"Unified PDF generated: {output_path}")
    return output_path

def initialize_resources():
    """Initialize resources and verify setup before accepting input."""
    global embeddings, vector_store_text, vector_store_image, llm, prompt

    # Verify Tesseract installation
    try:
        subprocess.run([pytesseract.tesseract_cmd, "--version"], capture_output=True, check=True)
        logger.info("Tesseract-OCR is installed and accessible.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Tesseract-OCR not found or inaccessible: {str(e)}. Please ensure Tesseract-OCR is installed and the path is correct.")
        sys.exit(1)

    # Verify PDF directory
    pdf_paths = list(PYTHONN1.pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.error(f"No PDFs found in {PYTHONN1.pdf_dir}")
        sys.exit(1)

    # Verify Excel file
    if not os.path.isfile(excel_file_path):
        logger.error(f"Excel file not found: {excel_file_path}")
        sys.exit(1)

    # Check existing material from PYTHONN1
    existing_components = PYTHONN1.check_existing_material()
    choice = 'rewrite'
    if existing_components:
        logger.info(f"Existing components found: {', '.join(existing_components)}")
        print(f"Existing components found: {', '.join(existing_components)}")
        print("Do you want to continue with existing data or rewrite it? (Enter 'continue' or 'rewrite'):")
        choice = input().strip().lower()
        while choice not in ['continue', 'rewrite']:
            print("Invalid input. Please enter 'continue' or 'rewrite':")
            choice = input().strip().lower()
    else:
        logger.warning("No existing components found. Will extract new data.")

    if choice == 'rewrite':
        logger.info("Rewriting data as requested or no existing components found.")
        selected_processes = ['text_extraction', 'image_extraction', 'text_index', 'image_index', 'inverted_index']
        PYTHONN1.clear_existing_files(selected_processes)
        extracted_text, image_chunks, global_keyword_index, image_metadata = PYTHONN1.batch_extract_pdfs(
            pdf_paths, PYTHONN1.image_dir, text_extraction_needed=True
        )
        if extracted_text:
            with open(PYTHONN1.text_output, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            with open(PYTHONN1.text_output, "rb") as f_in:
                with gzip.open(PYTHONN1.compressed_text, "wb") as f_out:
                    f_out.writelines(f_in)
        with open(PYTHONN1.inverted_index_file, "w") as f:
            json.dump(global_keyword_index, f)
        PYTHONN1.setup_vector_stores(
            extracted_text, image_metadata, PYTHONN1.faiss_text_index_dir, PYTHONN1.faiss_image_index_dir
        )
        existing_components = PYTHONN1.check_existing_material()

    # Load resources
    try:
        logger.info("Loading SentenceTransformer model.")
        embeddings = PYTHONN1.HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {str(e)}")
        sys.exit(1)

    if (PYTHONN1.faiss_text_index_dir / "index.faiss").exists():
        try:
            logger.info("Loading FAISS text index.")
            vector_store_text = PYTHONN1.FAISS.load_local(PYTHONN1.faiss_text_index_dir, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.warning(f"Failed to load text index: {str(e)}. Text similarity search will be skipped.")
            vector_store_text = None
    else:
        logger.warning("FAISS text index not found. Text similarity search will be skipped.")
        vector_store_text = None

    if (PYTHONN1.faiss_image_index_dir / "index.faiss").exists():
        try:
            logger.info("Loading FAISS image index.")
            vector_store_image = PYTHONN1.FAISS.load_local(PYTHONN1.faiss_image_index_dir, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.warning(f"Failed to load image index: {str(e)}. Image similarity search will be skipped.")
            vector_store_image = None
    else:
        logger.warning("FAISS image index not found. Image similarity search will be skipped.")
        vector_store_image = None

    try:
        logger.info("Setting up QA chain.")
        llm, prompt = PYTHONN1.setup_qa_chain()
    except Exception as e:
        logger.error(f"Failed to set up QA chain: {str(e)}")
        sys.exit(1)

    return existing_components

def process_keyword(keyword, existing_components):
    # Step 1: Process PDFs using PYTHONN1.py logic
    logger.info(f"Processing keyword '{keyword}' with PYTHONN1.py")
    pdf_paths = list(PYTHONN1.pdf_dir.glob("*.pdf"))

    # Load existing data
    logger.info("Using existing components from PYTHONN1.")
    try:
        with open(PYTHONN1.text_output, "r", encoding="utf-8") as f:
            extracted_text = f.read()
    except FileNotFoundError:
        logger.error(f"Text output file not found: {PYTHONN1.text_output}. Re-running extraction.")
        extracted_text, image_chunks, global_keyword_index, image_metadata = PYTHONN1.batch_extract_pdfs(
            pdf_paths, PYTHONN1.image_dir, keyword=keyword, text_extraction_needed=True
        )
        with open(PYTHONN1.text_output, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        with open(PYTHONN1.text_output, "rb") as f_in:
            with gzip.open(PYTHONN1.compressed_text, "wb") as f_out:
                f_out.writelines(f_in)

    try:
        with open(PYTHONN1.inverted_index_file, "r") as f:
            global_keyword_index = json.load(f)
    except FileNotFoundError:
        logger.error(f"Inverted index file not found: {PYTHONN1.inverted_index_file}. Re-running extraction.")
        extracted_text, image_chunks, global_keyword_index, image_metadata = PYTHONN1.batch_extract_pdfs(
            pdf_paths, PYTHONN1.image_dir, keyword=keyword, text_extraction_needed=True
        )
        with open(PYTHONN1.inverted_index_file, "w") as f:
            json.dump(global_keyword_index, f)

    image_chunks = {}
    image_metadata = []
    if PYTHONN1.image_dir.exists() and any(PYTHONN1.image_dir.glob("*_images.json")):
        for chunk_file in PYTHONN1.image_dir.glob("*_images.json"):
            pdf_name = chunk_file.stem.replace("_images", "") + ".pdf"
            try:
                with open(chunk_file, "r") as f:
                    chunk = json.load(f)
                    image_chunks[pdf_name] = chunk
                    for page_num, images in chunk.items():
                        for img in images:
                            proxy_text = f"{pdf_name} page {page_num} image: {img['path']}"
                            image_metadata.append({
                                "text": proxy_text,
                                "metadata": {
                                    "pdf_name": pdf_name,
                                    "page_num": int(page_num),
                                    "image_path": img['path']
                                }
                            })
            except Exception as e:
                logger.warning(f"Failed to load image chunk {chunk_file}: {str(e)}")
    else:
        logger.warning("No image chunks found. Skipping image metadata extraction.")

    # Perform QA
    keyword_matches = []
    for kw in global_keyword_index:
        if keyword.lower() in kw.lower():
            keyword_matches.extend(global_keyword_index[kw])

    text_context = ""
    if vector_store_text:
        try:
            docs_with_scores = vector_store_text.similarity_search_with_score(keyword, k=100)
            text_context = "\n".join([doc.page_content[:500] for doc, _ in docs_with_scores[:10]])
        except Exception as e:
            logger.warning(f"Text similarity search failed: {str(e)}. Using keyword matches only.")

    image_context = ""
    image_paths = []
    if vector_store_image:
        try:
            image_docs_with_scores = vector_store_image.similarity_search_with_score(keyword, k=50)
            image_context = "\n".join([doc.page_content[:500] for doc, _ in image_docs_with_scores[:5]])
            image_paths = [doc.metadata["image_path"] for doc, _ in image_docs_with_scores[:5]]
            for doc, _ in image_docs_with_scores:  # Fixed: Use image_docs_with_scores instead of docs_with_scores
                source = doc.metadata.get('source', 'unknown')
                if source == 'unknown' or not source.endswith('.pdf'):
                    continue
                page_match = re.search(r'\[Page (\d+)\]', doc.page_content)
                page_num = int(page_match.group(1)) if page_match else None
                chunk = PYTHONN1.load_image_chunk(source, PYTHONN1.image_dir)
                if page_num and str(page_num) in chunk:
                    image_paths.extend([img["path"] for img in chunk[str(page_num)]])
            image_paths = list(dict.fromkeys(image_paths))[:5]
        except Exception as e:
            logger.warning(f"Image similarity search failed: {str(e)}. Falling back to keyword matches.")
    else:
        for pdf_name, page_num, _ in keyword_matches:
            if not pdf_name.endswith('.pdf'):
                continue
            chunk = PYTHONN1.load_image_chunk(pdf_name, PYTHONN1.image_dir)
            if str(page_num) in chunk:
                image_paths.extend([img["path"] for img in chunk[str(page_num)]])
        image_paths = list(dict.fromkeys(image_paths))[:5]

    try:
        input_dict = {
            "text_context": text_context + "\nKeyword Matches: " + "; ".join(
                [f"{pdf}: {page}: {text}" for pdf, page, text in keyword_matches[:10]]),
            "image_context": image_context or "No image context available",
            "question": keyword
        }
        formatted_prompt = prompt.format(**input_dict)
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
        qa_answer = raw_answer.split("[/INST]")[-1].strip() if "[/INST]" in raw_answer else raw_answer.strip()
        if not qa_answer:
            qa_answer = "No relevant information found."
    except Exception as e:
        logger.error(f"Error generating QA answer: {str(e)}")
        qa_answer = f"Error generating answer: {str(e)}"

    # Step 2: Process Excel using PYTHONN2.py logic
    logger.info(f"Processing keyword '{keyword}' with PYTHONN2.py")
    try:
        df = pd.read_excel(excel_file_path)
        required_columns = ['Name', 'Description', 'Category', 'WORD IN KEYWORDS', 'WORD IN KEYWORDS 1']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in {excel_file_path}")
            products = pd.DataFrame()
        else:
            keyword_lower = keyword.lower()
            products = df[
                df['Name'].str.lower().str.contains(keyword_lower, na=False) |
                df['Description'].str.lower().str.contains(keyword_lower, na=False) |
                df['WORD IN KEYWORDS'].str.lower().str.contains(keyword_lower, na=False) |
                df['WORD IN KEYWORDS 1'].str.lower().str.contains(keyword_lower, na=False)
            ]
    except Exception as e:
        logger.error(f"Error reading Excel file: {str(e)}")
        products = pd.DataFrame()

    return qa_answer, keyword_matches, image_paths, products

def main():
    # Initialize resources upfront
    existing_components = initialize_resources()

    # Check for command-line argument
    keyword = None
    if len(sys.argv) > 1:
        keyword = sys.argv[1]
        sys.argv = [sys.argv[0]]

    while True:
        if not keyword:
            print("Ready to process keyword. Enter the keyword to search for (or 'exit' to quit):")
            keyword = input().strip()

        if keyword.lower() == 'exit':
            logger.info("Exiting the program.")
            break

        if not keyword:
            print("Please enter a valid keyword.")
            continue

        qa_answer, qa_keyword_matches, qa_image_paths, products = process_keyword(keyword, existing_components)
        if qa_answer or not products.empty:
            pdf_path = generate_unified_pdf(qa_answer, qa_keyword_matches, qa_image_paths, products, keyword, pdf_output_dir)
            print(f"Unified PDF generated: {pdf_path}")
        else:
            print(f"No results found for keyword '{keyword}'.")

        keyword = None

if __name__ == "__main__":
    main()