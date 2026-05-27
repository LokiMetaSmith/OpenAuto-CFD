#!/bin/bash
set -e

# Load Utility Functions if they exist, otherwise we just continue
if [ -f "lib/utils.sh" ]; then
    source lib/utils.sh
fi

if type print_header >/dev/null 2>&1; then
    print_header "Generating PDF Documentation"
else
    echo "=== Generating PDF Documentation ==="
fi

# Verify pandoc and pdflatex are installed
if ! command -v pandoc > /dev/null 2>&1; then
    echo "Error: pandoc is not installed."
    echo "Please install it via 'sudo apt-get install pandoc texlive-latex-base' or equivalent."
    (exit 1) || return 1
fi

if ! command -v pdflatex > /dev/null 2>&1; then
    echo "Error: pdflatex is not installed."
    echo "Please install it via 'sudo apt-get install texlive-latex-base' or equivalent."
    (exit 1) || return 1
fi

# Ensure template.tex exists
if [ ! -f "template.tex" ]; then
    echo "Error: template.tex not found."
    (exit 1) || return 1
fi

# Clean markdown of latex-breaking components
echo "Preprocessing markdown..."
cp TECHNICAL_REPORT.md temp_report.md
# Strip out tightlist before running Pandoc, if it gets added by standard markdown
sed -i 's/\\tightlist//g' temp_report.md

# Generate the PDF
echo "Running Pandoc conversion..."
# Use -f markdown-pipe_tables because longtable is fundamentally incompatible with twocolumn
if pandoc temp_report.md \
    -o publication.pdf \
    --pdf-engine=pdflatex \
    --template=template.tex \
    -f markdown-pipe_tables; then
    echo "Successfully generated publication.pdf"
else
    echo "Error: PDF generation failed."
    rm -f temp_report.md
    (exit 1) || return 1
fi

rm -f temp_report.md
