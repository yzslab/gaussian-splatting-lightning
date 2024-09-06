from pathlib import Path
from setuptools import setup, find_packages

setup_path = Path(__file__).parent
README = (setup_path / "README.md").read_text(encoding="utf-8")

with open("README.md", "r") as fh:
    long_description = fh.read()

def split_requirements(requirements):
    install_requires = []
    for requirement in requirements:
        if not requirement.startswith("-e"):
            install_requires.append(requirement)
    return install_requires

with open("./requirements.txt", "r") as f:
    requirements = f.read().splitlines()

install_requires = split_requirements(requirements)

setup(
    name = "gaussian-splatting-lightning",
    packages=find_packages(where="internal"),
    package_dir={'': 'internal'},
    description="A 3D Gaussian Splatting framework with various derived algorithms and an interactive web viewer",
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=install_requires
)
