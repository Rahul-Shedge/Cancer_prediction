from setuptools import setup

with open("README.md","r",encoding="utf-8") as f:
    long_description = f.read()

REPO_NAME = "Cancer_prediction"
AUTHOR_USER_NAME = "Rahul-Shedge"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = []

setup (
    name = SRC_REPO,
    version= "0.0.2",
    author= AUTHOR_USER_NAME,
    description= "MLOps workflow Cancer prediction",
    long_description= long_description,
    long_description_content_type = "test/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email= "rahulshedge555@gmail.com",
    packages=[SRC_REPO],
    license="MIT",
    py_modules= "=3.8.0",
    libraries=LIST_OF_REQUIREMENTS
)





