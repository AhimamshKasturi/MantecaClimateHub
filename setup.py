from setuptools import setup, find_packages

setup(
    name="manteca-climate-hub",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "sqlalchemy>=1.4.0",
        "requests>=2.28.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0"
    ],
    extras_require={
        "prophet": ["prophet>=1.1.0"],
        "arima": ["pmdarima>=2.0.0"]
    },
    python_requires=">=3.8",
)
