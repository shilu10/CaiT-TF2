rom setuptools import setup, find_packages


setup(
    name='cait',
    version='1.0',
    author='Shilash',
    description='CaiT transformer',
    python_requires='>=3.5, <4',
    packages=find_packages(include=['cait', 'cait.*']),
)