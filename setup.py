from setuptools import setup

# Read dependencies from requirements.txt
with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='plot_fatbands',
    version='0.1',
    py_modules=['plot_fatbands', 'plot_DOS'], 
    install_requires=required_packages,
)
