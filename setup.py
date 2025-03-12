from setuptools import find_packages, setup

setup(
    name="pydantic2django",
    version="0.1.0",
    description="Generate Django models from Pydantic models and convert between them",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/pydantic2django",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "django>=3.2",
        "pydantic>=2.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Framework :: Django",
        "Framework :: Django :: 3.2",
        "Framework :: Django :: 4.0",
        "Framework :: Django :: 4.1",
        "Framework :: Django :: 4.2",
    ],
)
