from setuptools import setup, find_packages

setup(
    name="NCAT",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "catboost",
        "category-encoders",
        "json"
    ],
    author="Junwon Yang",
    author_email="yangjunwon1309@kaist.ac.kr",
    description="A Python library for NCAT model implementation",
    url="https://github.com/yangjunwon1309/NCAT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
