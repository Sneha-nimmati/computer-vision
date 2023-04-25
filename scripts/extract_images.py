# from pdf2image import convert_from_path

# # Convert the PDF file to a list of PIL image objects
# pages = convert_from_path('../pdf/f941.pdf')

# # Loop through each page in the PDF file
# for page_num, page in enumerate(pages):
#     page.save("../data/sample_{}.png".format(page_num))


import os
from pdf2image import convert_from_path

# Path to the folder containing PDF files
pdf_folder = '../pdf/'
image_folder = "../data/"

# Loop through each file in the PDF folder
for filename in os.listdir(pdf_folder):
    # Check if the file is a PDF file
    if filename.endswith('.pdf'):
        # Generate the path to the PDF file
        pdf_path = os.path.join(pdf_folder, filename)

        # Convert the PDF file to a list of PIL image objects
        pages = convert_from_path(pdf_path)

        # Loop through each page in the PDF file
        for page_num, page in enumerate(pages):
            if page_num == 0:
                for iteration in range(0,333):
                    image_path = os.path.join(image_folder, f'{os.path.splitext(filename)[0]}_page{iteration}.png')
                    page.save(image_path)
