import os

from setuptools import setup, find_packages


def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(req_path, "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


setup(
    name="sonoclip",
    version="1.0",
    description="",
    author="OpenAI&ZeyiSun",
    packages=find_packages(exclude=["tests*"]),
    install_requires=read_requirements(),
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
