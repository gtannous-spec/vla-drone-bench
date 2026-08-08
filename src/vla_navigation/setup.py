import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'vla_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools', 'pyyaml', 'numpy'],
    zip_safe=True,
    maintainer='gtannous',
    maintainer_email='gtannous@example.com',
    description='Urban drone navigation benchmark for VLA research',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'classical_planner = vla_navigation.classical_planner:main',
            'vla_planner = vla_navigation.vla_planner:main',
        ],
    },
)
