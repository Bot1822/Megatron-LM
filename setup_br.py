import setuptools

# original default value that hard-coded in setup
version = "0.8.0"

try:
    from pathlib import Path
    this_dir = Path(__file__).parent
    version = (this_dir / 'version.txt').read_text()
except ImportError:
    try:
        with open('version.txt', 'r') as version_obj:
            version = version_obj.read()
    except IOError:
        pass
    except FileNotFoundError:
        pass

setuptools.setup(
    name="megatron",
    version=version,
    description="Official Megatron-LM package",
    packages=['megatron'],
    package_data={
        '': ['*'],
    },

    include_package_data=True,
    install_package_data=True,
    license='Apache2',
    license_file='./LICENSE',
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)