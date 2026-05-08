from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="User Survival Prediction with Astro Airflow , SQL , Redis , Grafana & Prometheus",
    version="0.1",
    author="Tannushree",
    packages=find_packages(),
    install_requires = requirements,
)