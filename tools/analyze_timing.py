#!/usr/bin/env python3
# 사용: python3 tools/analyze_timing.py logs/<시각>_<tag>
# 타이밍 CSV 분석: 노드/구간별 통계, phase별 틱 지터, driver→arm 틱 전달지연, 비전 검출률/밝기, 카메라 dup
import sys, os
import numpy as np
import pandas as pd

D = sys.argv[1]
pd.set_option("display.width", 200)


def load(n):
    df = pd.read_csv(os.path.join(D, f"{n}.csv"), keep_default_na=False)
    df["t"] = (df.t_ros_ns - df.t_ros_ns.min()) / 1e9
    return df


def stats(df):
    g = df.groupby("section").dur_ms
    return pd.DataFrame({"n": g.size(), "mean": g.mean(), "std": g.std(), "p50": g.median(),
                         "p95": g.quantile(.95), "p99": g.quantile(.99), "max": g.max()}).round(2)


def kv(note, key):
    for part in note.split(";"):
        if part.startswith(key + "="):
            return part[len(key) + 1:]
    return None


drv, arm, vis, cam = (load(n) for n in ("piper_driver_node", "arm_node", "vision_node", "piper_eih_camera_node"))
t0 = min(d.t_ros_ns.min() for d in (drv, arm, vis, cam))
dur = (max(d.t_ros_ns.max() for d in (drv, arm, vis, cam)) - t0) / 1e9
print(f"run {D}  duration {dur:.1f}s\n")
for name, d in (("driver", drv), ("arm", arm), ("vision", vis), ("camera", cam)):
    print(f"== {name}\n{stats(d)}\n")

# driver: phase별 interval 지터 + 송신
tt = drv[drv.section == "tick.total"].copy()
tt["phase"] = tt.note.str.split(";").str[0]
tt["sent"] = tt.note.map(lambda s: kv(s, "sent")).astype(int)
tt["tick"] = tt.note.map(lambda s: kv(s, "tick")).astype(int)
iv = drv[drv.section == "tick.interval"].set_index("t_ros_ns").dur_ms
tt["interval"] = tt.t_ros_ns.map(iv)
ph = tt.groupby("phase", sort=False).agg(ticks=("tick", "size"), t_start=("t", "min"), t_end=("t", "max"),
                                          sent=("sent", "sum"), iv_mean=("interval", "mean"),
                                          iv_std=("interval", "std"), iv_max=("interval", "max"),
                                          total_p95=("dur_ms", lambda x: x.quantile(.95)))
print("== driver by phase (interval ms, total ms)\n", ph.round(2), "\n")
late = (iv > 16.667 * 1.5).sum()
print(f"driver tick interval > 25ms: {late}/{len(iv)} ({100*late/len(iv):.2f}%)  "
      f"mean Hz={1000/iv.mean():.2f}\n")

# driver→arm /plant/tick 전달지연: 같은 tick 번호의 ROS time 차이
at = arm[arm.section == "tick.total"].copy()
at["tick"] = at.note.map(lambda s: kv(s, "tick")).astype(int)
m = at.merge(tt[["tick", "t_ros_ns"]], on="tick", suffixes=("_arm", "_drv"))
lat = (m.t_ros_ns_arm - m.t_ros_ns_drv) / 1e6
print(f"== /plant/tick driver→arm delay ms: n={len(lat)} mean={lat.mean():.2f} p50={lat.median():.2f} "
      f"p95={lat.quantile(.95):.2f} p99={lat.quantile(.99):.2f} max={lat.max():.2f}")
print(f"   ticks lost (driver ticks not seen by arm): {len(tt) - len(m)}\n")
trans = at[at.note.str.contains(";to=")]
print("== arm phase transitions (t s, step)")
for _, r in trans.iterrows():
    print(f"   {r.t:7.2f}s  {r.note}")
print()

# vision: 결과별 건수, 밝기, phase 매핑
vt = vis[vis.section == "eih.total"].copy()
vt["pick"] = vt.note.map(lambda s: kv(s, "pick") or ("no_tf" if s == "no_tf" else s))
vt["lum"] = pd.to_numeric(vt.note.map(lambda s: kv(s, "lum")), errors="coerce")
print("== vision outcome counts\n", vt.pick.value_counts().to_string(), "\n")
print(f"   lum mean={vt.lum.mean():.1f} std={vt.lum.std():.1f} min={vt.lum.min()} max={vt.lum.max()}")
okv = vt[vt.pick == "ok"]
print(f"   total ms | ok: mean={okv.dur_ms.mean():.2f} p95={okv.dur_ms.quantile(.95):.2f} max={okv.dur_ms.max():.2f}"
      f" | all: mean={vt.dur_ms.mean():.2f} max={vt.dur_ms.max():.2f}")
sd = vis[vis.section == "eih.stamp_to_done"].dur_ms
print(f"   stamp_to_done ms: mean={sd.mean():.2f} p95={sd.quantile(.95):.2f} max={sd.max():.2f}\n")

# camera: 실제 fps, dup, cap_to_pub
ci = cam[cam.section == "capture.interval"].dur_ms
pt = cam[cam.section == "pub.total"]
dup = pt.note.map(lambda s: kv(s, "dup")).astype(int).sum()
c2p = cam[cam.section == "cap_to_pub"] if "cap_to_pub" in cam.section.values else cam[cam.section == "pub.cap_to_pub"]
print(f"== camera capture fps={1000/ci.mean():.2f} interval std={ci.std():.2f} max={ci.max():.2f}  "
      f"pub dup={dup}/{len(pt)}  cap_to_pub mean={c2p.dur_ms.mean():.1f} p95={c2p.dur_ms.quantile(.95):.1f} max={c2p.dur_ms.max():.1f}")
# 영상 획득~비전 처리완료 end-to-end (평균 합)
print(f"   frame age at vision done ≈ cap_to_pub({c2p.dur_ms.mean():.1f}) + stamp_to_done({sd.mean():.1f}) "
      f"= {c2p.dur_ms.mean() + sd.mean():.1f} ms (mean)")
