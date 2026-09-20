from setuptools import setup

package_name = "piper_hw_pkg"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch",
         ["launch/piper_real.launch.py", "launch/eih_debug_view.launch.py",
          "launch/rsp_only.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="da",
    maintainer_email="ekgks3451@gmail.com",
    description="실물 PiPER CAN 구동 브릿지(piper_sdk) + 손목캠 소스",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "piper_driver_node = piper_hw_pkg.piper_driver_node:main",
            "piper_gripper_node = piper_hw_pkg.piper_gripper_node:main",
            "piper_eih_camera_node = piper_hw_pkg.piper_eih_camera_node:main",
            "piper_fake_amr_node = piper_hw_pkg.piper_fake_amr_node:main",
            "eih_marker_debug_node = piper_hw_pkg.eih_marker_debug_node:main",
        ],
    },
)
