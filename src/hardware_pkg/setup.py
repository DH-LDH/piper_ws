from setuptools import setup

package_name = "hardware_pkg"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="da",
    maintainer_email="ekgks3451@gmail.com",
    description="STEP22 그리퍼 노드 (α-SMC 힘제어)",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gripper_node = hardware_pkg.gripper_node:main",
        ],
    },
)
