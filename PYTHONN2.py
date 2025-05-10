import os
import sys
import re
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# Define paths
excel_file_path = r"D:\ITO\DENTAL KEYWORDS\keyword2\Output_With_Image_Links.xlsx"
output_directory = r"D:\ITO\DENTAL KEYWORDS\keyword2\qa_pdfs"


def sanitize_filename(keyword):
    keyword = keyword.lower()
    keyword = re.sub(r'[^a-z0-9]', '_', keyword)
    keyword = re.sub(r'_+', '_', keyword)
    return keyword.strip('_')


def create_header(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 10)
    canvas.drawString(30, doc.pagesize[1] - 30, f"Search Report for Keyword: {doc.keyword}")
    canvas.drawString(doc.pagesize[0] - 100, doc.pagesize[1] - 30, f"Page {doc.page}")
    canvas.restoreState()


def create_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    canvas.drawString(30, 30, f"Generated on: {timestamp}")
    canvas.restoreState()


def generate_pdf_content(products, keyword, output_dir, pdf_name):
    output_path = os.path.join(output_dir, f"{pdf_name}.pdf")
    doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=50, bottomMargin=50)
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
    story.append(Paragraph(f"Product Search Report", title_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(f"Keyword: {keyword}", heading_style))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", heading_style))
    story.append(Spacer(1, 1 * inch))

    # Matching Products
    story.append(Paragraph("Matching Products", heading_style))
    story.append(Spacer(1, 0.25 * inch))

    failed_images = []

    for index, row in products.iterrows():
        name = str(row['Name']) if not pd.isna(row['Name']) else "N/A"
        category = str(row['Category']) if not pd.isna(row['Category']) else "N/A"
        description = str(row['Description']) if not pd.isna(row['Description']) else "N/A"
        price = str(row['Price']) if not pd.isna(row['Price']) else "N/A"
        link = str(row['Link']) if not pd.isna(row['Link']) else "N/A"
        features = str(row['Features']) if not pd.isna(row['Features']) else "N/A"

        # Product Details Table
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

        # Images Table
        image_columns = [f'Image Path {i}' for i in range(1, 11)]
        valid_images = []
        for col in image_columns:
            if col in row and not pd.isna(row[col]):
                path = str(row[col]).strip()
                if os.path.isfile(path) and path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img = Image(path)
                        img.drawHeight = 1.5 * inch
                        img.drawWidth = 1.5 * inch
                        valid_images.append(img)
                    except Exception as e:
                        failed_images.append((path, str(e)))
                else:
                    failed_images.append((path, "File not found or invalid format"))

        if valid_images:
            # Create rows of 3 images each
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

    # Image Inclusion Status
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
    print(f"PDF generated successfully: {output_path}")


def search_and_generate_pdf(excel_path, keyword, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"Excel file not found: {excel_path}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    required_columns = ['Name', 'Description', 'Category', 'WORD IN KEYWORDS', 'WORD IN KEYWORDS 1']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {', '.join(missing_columns)}")
        return

    keyword_lower = keyword.lower()
    matches = df[
        df['Name'].str.lower().str.contains(keyword_lower, na=False) |
        df['Description'].str.lower().str.contains(keyword_lower, na=False) |
        df['WORD IN KEYWORDS'].str.lower().str.contains(keyword_lower, na=False) |
        df['WORD IN KEYWORDS 1'].str.lower().str.contains(keyword_lower, na=False)
        ]

    if matches.empty:
        print(f"No products found for keyword '{keyword}'.")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_name = f"{sanitize_filename(keyword)}_{timestamp}"
    generate_pdf_content(matches, keyword, output_dir, pdf_name)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        keyword = sys.argv[1]
        sys.argv = [sys.argv[0]]
    else:
        keyword = None

    while True:
        if not keyword:
            keyword = input("Enter the keyword to search for (or 'exit' to quit): ").strip()

        if keyword.lower() == 'exit':
            print("Exiting the program.")
            break

        if not keyword:
            print("Please enter a valid keyword.")
            continue

        search_and_generate_pdf(excel_file_path, keyword, output_directory)
        keyword = None