"""Activity A: Convert PDFs to markdown."""

from pathlib import Path

import pymupdf4llm


def convert_pdfs_to_markdown(pdf_dir: str, output_dir: str) -> None:
    """Convert all PDFs in pdf_dir to markdown files in output_dir.

    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save markdown files

    """
    pdf_path = Path(pdf_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        print("Please add your project PDF reports to data/project_reports/")
        return

    print(f"Found {len(pdf_files)} PDF files to convert:")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")

        try:
            # Convert PDF to markdown using pymupdf4llm
            md_text = pymupdf4llm.to_markdown(str(pdf_file))

            # Create output filename
            output_file = output_path / f"{pdf_file.stem}.md"

            # Save markdown
            output_file.write_text(md_text, encoding="utf-8")

            # Report statistics
            lines = md_text.split("\n")
            headers = [line for line in lines if line.startswith("#")]

            print("  ✓ Converted successfully")
            print(f"  - Output: {output_file}")
            print(f"  - Size: {len(md_text):,} characters")
            print(f"  - Sections detected: {len(headers)}")

        except Exception as e:
            print(f"  ✗ Error converting {pdf_file.name}: {e}")

    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"Markdown files saved to: {output_dir}")
    print(f"{'=' * 60}")


def main() -> None:
    """Convert PDFs to markdown format."""
    pdf_dir = "data/project_reports"
    output_dir = "outputs/markdown"

    print("PDF to Markdown Converter")
    print("=" * 60)

    convert_pdfs_to_markdown(pdf_dir, output_dir)


if __name__ == "__main__":
    main()
