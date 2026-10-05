from setuptools import setup, find_packages

setup(
    name="cascade-pid",
    version="0.1.0",
    description="Confidence-based two-stage cascade for prompt-injection detection",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "jsonlines>=4.0.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ],
    extras_require={
        "train": [
            "torch>=2.1.0",
            "transformers>=4.40.0",
            "tokenizers>=0.15.0",
            "datasets>=2.19.0",
            "peft>=0.11.0",
            "trl>=0.8.0",
            "bitsandbytes>=0.43.0",
            "accelerate>=0.30.0",
            "wandb>=0.17.0",
        ],
        "test": ["pytest>=8.0.0"],
    },
)
