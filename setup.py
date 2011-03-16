from setuptools import setup

setup(
    name = "ScanRadSim",
    version = "0.0.1",
    author = "Benjamin Root",
    author_email = "ben.v.root@gmail.com",
    description = "System for simulating the scanning by a radar, with a focus on analyzing adaptive sensing methods",
    license = "BSD",
    keywords = ("radar", "simulator", "scanning", "scheduling", "adaptive sensing"),
    url = "https://github.com/WeatherGod/ScanRadSim",
    packages = ['ScanRadSim',],
    package_dir = {'': 'lib'}
    )

