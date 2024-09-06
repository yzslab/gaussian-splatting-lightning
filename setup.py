from pathlib import Path
from setuptools import setup, find_packages

setup_path = Path(__file__).parent
README = (setup_path / "README.md").read_text(encoding="utf-8")

with open("README.md", "r") as fh:
    long_description = fh.read()

def split_requirements(requirements):
    install_requires = []
    dependency_links = []
    for requirement in requirements:
            install_requires.append(requirement)

    return install_requires, dependency_links

def load_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

install_requires = load_requirements("requirements.txt")

setup(
    name = "gaussian-splatting-lightning",
    packages=find_packages(where="internal"),
    package_dir={'': 'internal'},
    description="A 3D Gaussian Splatting framework with various derived algorithms and an interactive web viewer",
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    python_requires=">=3.7",
)
