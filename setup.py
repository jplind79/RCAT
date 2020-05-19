import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rcat-jplind79", # Replace with your own username
    version="1.0.1",
    author="Petter Lind",
    author_email="petter.lind@smhi.se",
    description="Regional Climate Analysis Tool (RCAT)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jplind79/rcat",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
