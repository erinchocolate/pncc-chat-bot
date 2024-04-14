import os
import subprocess
from dotenv import load_dotenv
from llama_index.llama_pack import download_llama_pack

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

EmbeddedTablesUnstructuredRetrieverPack = download_llama_pack(
    "EmbeddedTablesUnstructuredRetrieverPack",
    "./embedded_tables_unstructured_pack",
)

def convert_pdf_to_html(pdf_path, html_path):
    command = f"pdf2htmlEX {pdf_path} --dest-dir {html_path}"
    subprocess.call(command, shell=True)

input_pdf = "quarterly-nvidia.pdf"
output_pdf = "quarterly-nvidia"

convert_pdf_to_html(input_pdf, output_pdf)

