# plant_node는 isaacsim/omni/pxr에 묶여 있어 시스템 python(ros2 run이 쓰는 것)이
# 아니라 Isaac Sim 번들 python.sh 위에서만 돌 수 있다. 이 파일은 아무 무거운
# import도 하지 않는 얇은 런처로, python.sh로 프로세스를 통째로 교체(exec)해서
# 실제 구현(plant_node.py, 같은 디렉터리)을 그 안에서 실행한다.
# ros2 run으로 넘어온 인자(--keep-open, --max-sec 등)는 그대로 전달된다.
import os
import sys

ISAACSIM_DIR = "/home/da/isaacsim"
PYTHON_SH = os.path.join(ISAACSIM_DIR, "python.sh")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    real_script = os.path.join(here, "plant_node.py")
    if not os.path.exists(PYTHON_SH):
        print(f"★ Isaac Sim python.sh를 찾을 수 없습니다: {PYTHON_SH}", file=sys.stderr)
        sys.exit(1)
    os.execv(PYTHON_SH, [PYTHON_SH, real_script] + sys.argv[1:])


if __name__ == "__main__":
    main()
