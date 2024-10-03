import os

from setuptools import find_packages, setup


def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def find_nested_items(base_path, prefix):
    items = set()
    for root, _, files in os.walk(base_path):
        rel_path = os.path.relpath(root, base_path)
        if rel_path != ".":
            items.add(f"{prefix}.{rel_path.replace(os.path.sep, '.')}")
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                module_path = os.path.join(rel_path, os.path.splitext(file)[0])
                if module_path != ".":
                    items.add(f"{prefix}.{module_path.replace(os.path.sep, '.')}")
    return list(items)


def get_packages_and_modules():
    packages = {"ecole_mo9_demo"}
    py_modules = set()
    for item in os.listdir("src"):
        item_path = os.path.join("src", item)
        if os.path.isdir(item_path):
            nested_items = find_nested_items(item_path, f"ecole_mo9_demo.{item}")
            packages.update(i for i in nested_items if i.count(".") == 2)
            py_modules.update(i for i in nested_items if i.count(".") > 2)
        elif item.endswith(".py") and item != "__init__.py":
            py_modules.add(f"ecole_mo9_demo.{os.path.splitext(item)[0]}")
    return list(packages), list(py_modules)


packages, py_modules = get_packages_and_modules()

setup(
    name="ecole_mo9_demo",
    version="0.1.0",
    author="Ansel Blume",
    author_email="blume5@illinois.edu",
    description="demo package for ecole_mo9_demo",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    package_dir={"ecole_mo9_demo": "src"},
    packages=packages,
    py_modules=py_modules,
    package_data={"ecole_mo9_demo": ["**/data/*", "**/*.json", "**/*.pth"]},
    include_package_data=True,
    install_requires=read_file("requirements.txt").splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
