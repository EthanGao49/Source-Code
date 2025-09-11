from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core requirements (excluding optional AI dependencies)
core_requirements = [
    "yfinance>=0.2.18",
    "pandas>=1.5.0", 
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    "scipy>=1.9.0"
]

# AI/ML requirements (optional)
ai_requirements = [
    "tensorflow>=2.10.0",
    "scikit-learn>=1.1.0"
]

setup(
    name="qbt",
    version="0.1.0",
    author="QBT Development Team",
    description="A Python library for quantitative backtesting of trading strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "ai": ai_requirements,
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
        "all": ai_requirements + [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0", 
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qbt-example=qbt.examples.run_example:main",
            "qbt-ai-example=qbt.examples.ai_signals_example:main",
        ],
    },
)