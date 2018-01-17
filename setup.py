import setuptools

setuptools.setup(
    author="Victor Kulikov",
    author_email="kulikov.victor@gmail.com",
    install_requires=[
        "numpy",
        "skimage",
        "sklearn",
        "matplotlib",
        "scikit",
        "matplotlib_scalebar",
        "torch"
    ],
    package_data={
        "dognet"
    }
    license="GNU v3",
    name="dognet",
    url="https://github.com/kulikovv/dognet",
    version="0.0.1"
)
