from pathlib import Path
from setuptools import setup

setup(
    name='krr_opt',
    version='0.1.0',
    license='MIT',  # https://choosealicense.com/licenses/mit/
    packages=[
        'krr_opt',
    ],
    author='Artem Kokorin',
    author_email='artem.kokorin.001@student.uni.lu',
    python_requires='>=3.9',
    # package_data={},  # for non-python files inside the package
    # include_package_data=True,
    install_requires=[
        # 'shgo==0.5.1',
        'joblib>=1.3.0',
        'scipy>=1.11.0',
        'numpy>1.23.0,<1.24.0',  # TODO 1.24 and above is incompatible with QML, but remove that for further development
        'matplotlib>=3.7.0',
        'scikit-learn>=1.5.0'
    ],
    long_description=(Path(__file__).parent / 'README.md').open().read()
)
