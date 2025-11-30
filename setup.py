from setuptools import setup, find_packages

setup(
    name="fact_clip",
    version="1.0.0",
    description="FACT with CLIP for Zero-Shot Action Segmentation",
    author="Leonardo Thomaz",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy",
        "scipy",
        "yacs",
        "wandb",
        "tqdm",
        "transformers>=4.30.0",
    ],
    extras_require={
        "viz": [
            "matplotlib",
            "umap-learn",
            "scikit-learn",
        ],
    },
)

