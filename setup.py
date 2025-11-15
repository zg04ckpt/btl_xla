from setuptools import setup, find_packages

setup(
    name='handwritten-digit-recognition',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow>=2.10',
        'opencv-python>=4.5',
        'numpy',
        'Pillow',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
            'handwritten-digit-recognition=main:main',  # Adjust the entry point as necessary
        ],
    },
    include_package_data=True,
    description='A desktop application for recognizing handwritten digits from images.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/handwritten-digit-recognition',  # Update with your repository URL
    license='MIT',
)