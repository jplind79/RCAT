import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RCAT",
    version="1.1",
    author="Petter Lind",
    author_email="petter.lind@smhi.se",
    description="Regional Climate Analysis Tool (RCAT)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jplind79/RCAT",
    keywords=["python", "climate", "model", "interpolation", "plotting", 
              "plots", "dask", "diagnostic", "science", "numpy"],
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    install_requires=['numpy', 'xarray', 'esmpy', 'xesmf', 'dask[complete]', 
                      'netcdf4', 'dask-jobqueue'],
)
