from setuptools import setup

package_name = "plant_pkg"

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
    description="STEP22 Isaac Sim 플랜트(시뮬레이션 호스트) 노드",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "plant_node = plant_pkg.plant_launcher:main",
        ],
    },
)
