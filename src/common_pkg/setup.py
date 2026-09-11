from setuptools import setup

package_name = "common_pkg"

setup(
    name=package_name,
    version="0.0.1",
    py_modules=["step22_common"],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="da",
    maintainer_email="ekgks3451@gmail.com",
    description="STEP22 공용 상수/순수수학 헬퍼(step22_common)",
    license="TODO",
    tests_require=["pytest"],
)
