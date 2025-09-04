#!/usr/bin/env python

import fpdf
from fpdf import FPDF
from pathlib import Path

# fpdf.set_global('SYSTEM_TTFONTS', str(Path('~/Library/Fonts').expanduser()))

# Sample invoice data
invoice_data = {
    "logo": "path/to/logo.png",
    "company_name": "Example Inc.",
    "invoice_number": "INV001",
    "date": "2023-04-01",
    "items": [
        {"name": "Item 1", "quantity": 2, "price": 10.99},
        {"name": "Item 2", "quantity": 1, "price": 5.99}
    ]
}

# Create PDF
pdf = FPDF(orientation="landscape", format="A4")
print(pdf.eph, pdf.epw)

pdf.add_page()
# pdf.add_font("Raleway-Medium", "", fname="Raleway-Medium.ttf", uni=True)
# pdf.add_font("Raleway-SemiBold", "", fname="Raleway-SemiBold.ttf", uni=True)
# pdf.set_font("Raleway-Medium", size=12)

# # Add header with logo and company name
# # pdf.image(invoice_data["logo"], x=10, y=10, w=50)
# pdf.cell(200, 10, txt=invoice_data["company_name"], ln=True, align="C")

# # Add invoice details
# pdf.cell(200, 10, txt=f"Invoice Number: {invoice_data['invoice_number']}", ln=True, align="L")
# pdf.cell(200, 10, txt=f"Date: {invoice_data['date']}", ln=True, align="L")

# # Create table for items
# pdf.ln(10)  # Add a line break
# pdf.set_font("Raleway-SemiBold", size=10)
# pdf.cell(30, 10, txt="Item", border=1, align="C", fill=False)
# pdf.cell(50, 10, txt="Quantity", border=1, align="C", fill=False)
# pdf.cell(50, 10, txt="Price", border=1, align="C", fill=False)
# pdf.ln(10)
# pdf.set_font("Raleway-Medium", size=10)
# for item in invoice_data["items"]:
#     pdf.cell(30, 10, txt=item["name"], border=1, align="L", fill=False)
#     pdf.cell(50, 10, txt=str(item["quantity"]), border=1, align="C", fill=False)
#     pdf.cell(50, 10, txt=str(item["price"]), border=1, align="R", fill=False)
#     pdf.ln(10)

import matplotlib.pyplot as plt
from io import BytesIO


pdf.add_font('Raleway', '', '/Users/ale/Library/Fonts/Raleway-Regular.ttf')
pdf.add_font('Raleway', 'B', '/Users/ale/Library/Fonts/Raleway-Bold.ttf')
pdf.set_margins(0., 0.)
fig, ax = plt.subplots(figsize=(5,5))

ax.plot([1,2,3],[6,5,4],antialiased=True,linewidth=2,color='red',label='a curve')
ax.set_title(r'$\omega$')

img = BytesIO()
fig.savefig(img,format='svg')

pdf.image(img, h=pdf.eph, w=pdf.eph)
pdf.set_font("Raleway", size=12)
pdf.set_xy(pdf.eph-10, 20)
pdf.write_html("<b><h2>Hello</h2></b><br/>aaaaa")

# import IPython
# IPython.embed(colors='neutral')
pdf.output("fpdf_example.pdf")