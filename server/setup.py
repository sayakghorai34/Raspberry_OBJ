from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="camera-inference-server",
    version="0.1.0",
    packages=find_packages(where="src/main/python"),
    package_dir={"": "src/main/python"},
    python_requires=">=3.11",
    install_requires=required,
    entry_points={
        'console_scripts': [
            'camera-inference-server=camera_inference.scripts.run_server:main',
        ],
    },
    author="Sayak ghorai",
    #author_email="your.email@example.com",
    description="A camera streaming server with real-time object detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="camera, object detection, yolo, streaming",
    #url="https://github.com/yourusername/camera-inference-server",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
)
