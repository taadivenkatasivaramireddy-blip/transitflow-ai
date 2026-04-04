from setuptools import setup, find_packages

setup(
    name="takshashila-transit-openenv",
    version="1.0.0",
    description="Real-world college bus fleet management RL environment (OpenEnv spec)",
    author="Ram",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0",
    ],
    extras_require={
        "ui": ["gradio>=4.0", "pandas>=2.0", "plotly>=5.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
