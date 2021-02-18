from setuptools import find_packages, setup

setup(
    name="ml_example",
    packages=find_packages(),
    version="0.1.0",
    description="Example of ml project",
    author="Your name (or your organization/company/team)",
    entry_points={
        "console_scripts": [
            "ml_example_train = ml_example.train_pipeline:train_pipeline_command"
        ]
    },
    install_requires=[
        "click",
        "Sphinx",
        "coverage",
        "awscli",
        "flake8",
        "setuptools",
        "python-dotenv>=0.5.1",
        "pytest",
        "scikit-learn",
        "dataclasses==0.8",
        "pyyaml==3.11",
        "marshmallow-dataclass==8.3.0",
        "pandas",
    ],
    license="MIT",
)
