
"""Setup script for predictive maintenance package."""
=======
"""Setup configuration for predictive-maintenance toolkit."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

=======
setup(
    name="predictive-maintenance",
    version="0.1.0",
    author="Predictive Maintenance Team",

    description="Machine learning models and pipelines for equipment health monitoring and failure prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manojsai7/predictive-maintenance",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
=======
    description="A toolkit for forecasting equipment health, catching issues early, and reducing downtime with ML-driven insights",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",

        "License :: OSI Approved :: MIT License",
=======

        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",

    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0",
            "flake8>=4.0.0",
        ],
    },
=======
        "Programming Language :: Python :: 3.11",
    ],

)
