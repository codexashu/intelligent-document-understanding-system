from setuptools import setup, find_packages

setup(
    name="intelligent-document-understanding-system",
    version="1.0.0",
    author="Ashutosh Pandey",
    author_email="pandeyashutosh082007@gmail.com",
    description="End-to-End Deep Learning System for Document Layout Analysis, KIE, Table Extraction, and DocVQA",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.5.0",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "python-multipart>=0.0.6",
    ],
)
