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
    description="로봇팔 픽앤플레이스 시퀀스 제어(arm_node)",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "arm_node = control_pkg.arm_node:main",
        ],
    },
)
