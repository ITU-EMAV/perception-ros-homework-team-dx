from setuptools import find_packages, setup
from glob       import glob
import os

package_name = 'perception_ros'


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,'Perception','Perception.model','Perception.utils'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='emav',
    maintainer_email='emav@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'perception = perception_ros.perception:main',
        ],
    },
)
