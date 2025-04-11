from setuptools import setup, find_packages
import versioneer

setup(
    name='ccm_tool',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Convergent Connectivity Mapping Tool',
    author='Amin Saberi',
    author_email='amnsbr@gmail.com',
    url='https://github.com/amnsbr/ccm_tool',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'NiMARE',
        'nilearn',
        'boto3',
        'tqdm',
        'seaborn',
        'statsmodels',
    ],
)