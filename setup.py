from setuptools import setup


setup(
    name='cait'
    version='1.0',
    author='Shilash',
    description='CaiT transformer',
    python_requires='>=3.5, <4',
    packages=find_packages(include=['cait', 'cait.*']),
    install_requires=[
        'PyYAML',
        'pandas==0.23.3',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0,,
        'tensorflow>2'
    ]
)