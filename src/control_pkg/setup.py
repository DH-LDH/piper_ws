from setuptools import setup

package_name = "control_pkg"

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
    description="STEP22 제어 노드 (AMR 추종 + 로봇팔 시퀀스)",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "amr_node = control_pkg.amr_node:main",
            "arm_node = control_pkg.arm_node:main",
        ],
    },
)
