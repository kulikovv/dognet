import setuptools

setuptools.setup(
    author="Victor Kulikov",
    author_email="kulikov.victor@gmail.com",
    install_requires=[
        "numpy",
        "scikit-image",
        "sklearn",
        "matplotlib",
        "scipy",
        "matplotlib_scalebar",
        "torch"
    ],
    packages=setuptools.find_packages(
        exclude=[
            "notebooks",
	    "images",
	    "datasets"
        ]
    ),
    license="GNU v3",
    name="dognet",
    url="https://github.com/kulikovv/dognet",
    version="0.0.1"
)
