#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive 3D coverage viewer — original planning algorithm preserved.
依赖: numpy, matplotlib；交互窗口需本机可用的 Tk 或 Qt 后端。
运行: python Coverage_Path_Planner_3D_interactive.py --shape F
全部形状: --shape all（关闭当前窗口后进入下一种形状）
无窗口终态导出: --headless；截图保存至 --output 指定目录。
窗口在原规划与补漏完成后打开，实时逐步回放，不是在线重规划。
空格暂停/继续；右箭头单步；鼠标拖动旋转、滚轮缩放；R复位；S截图。
轮廓由原ASCII单元整体刚性旋转；覆盖率保持原离散采样模型。
"""

# ============================ 导入 ============================
import math, heapq, numpy as np
from collections import deque
import matplotlib
# 默认使用本机 GUI 后端；仅显式无窗口模式使用 Agg。
import sys
if "--headless" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import animation
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ====================== 全局参数 / 地图 =======================
GRID        = (60, 60)          # 栅格大小 (W,H)
YAW_DIV     = 16                # 朝向离散份数（越大越细）
RES_YAW     = 2 * math.pi / YAW_DIV

# 硬安全距离（以格为单位）。会用于地图“欧式膨胀”。
# 注：覆盖规划要求机器人（占多个格子）在“固定朝向”下也能连通到地图各处，
# 否则一大块区域会因为“转不过身”而覆盖不到。这里把膨胀距离调小一些
# （原点到点规划版本里是 1.0），配合下面调低的障碍密度，
# 使得机器人在固定朝向下的自由位形空间基本连通（见脚本末尾自检打印）。
CLEARANCE_CELLS = 0.75

# 覆盖策略："shortest" = 集合覆盖+近似TSP，总轨迹更短；"boustrophedon" = 弓字形来回扫
COVERAGE_STRATEGY = "shortest"

# 覆盖规划参数
COVER_YAW_DEG   = 0.0     # 弓字形模式下固定使用的朝向（沿 x 方向来回扫）
ROW_OVERLAP     = 0.25    # 行间重叠比例（0~1），越大越不容易漏扫，但行数更多
MAX_ANIM_FRAMES = 260     # 动图最多渲染的帧数（覆盖计算本身仍是逐格精确的，不受此限制）

# 测试的机器人形状（依次跑一遍，分别出一张全覆盖动图）
SHAPES_TO_RUN = ["T", "A", "F"]

# ----------------------- 构造地图 + 障碍 ----------------------
W, H = GRID
MAP  = np.zeros(GRID, dtype=np.uint8)

# 随机“噪点风格”障碍。密度比原点到点版本（3%）低（1%），
# 原因同上：覆盖任务要求机器人本体（多格、非点状）在固定朝向下
# 也能连通地图的大部分区域，否则“全覆盖”无从谈起。
np.random.seed(42)
num_obstacles = int(W * H * 0.01)
rnd_indices = np.random.choice(W * H, num_obstacles, replace=False)
MAP[np.unravel_index(rnd_indices, GRID)] = 1





INFLATED_MAP = inflate_map_euclidean(MAP, CLEARANCE_CELLS)

# ======================== 形状与姿态 ==========================
def ascii_to_offsets(lines, on="#", center=None):
    if not lines:
        return []
    Hh = len(lines)
    Ww = max(len(r) for r in lines)
    norm = [r.ljust(Ww, " ") for r in lines]
    if center is None:
        cx, cy = (Ww - 1) / 2.0, (Hh - 1) / 2.0
    else:
        cx, cy = center
    out = []
    for iy, row in enumerate(norm):
        for ix, ch in enumerate(row):
            if ch == on:
                out.append((ix - cx, iy - cy))
    return out

T_ASCII = [
    "  #  ",
    "  #  ",
    "#####",
]
A_ASCII = [
    "#   #",
    "#   #",
    "#####",
    " # # ",
    "  #  ",
]
F_ASCII = [
    "#    ",
    "#    ",
    "#####",
    "#    ",
    "#####",
]

SHAPES = {
    "T": ascii_to_offsets(T_ASCII),
    "A": ascii_to_offsets(A_ASCII),
    "F": ascii_to_offsets(F_ASCII),
}

def deg2rad(d): return d * math.pi / 180.0
def yaw_bin(yaw): return int(round((yaw % (2*math.pi)) / RES_YAW)) % YAW_DIV
def bin_to_yaw(k): return (k % YAW_DIV) * RES_YAW

# ====================== 碰撞检测与几何 =======================
def shortest_angle_diff(a, b):
    return (b - a + math.pi) % (2*math.pi) - math.pi

def rotated_footprint(yaw, base):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(c*x - s*y, s*x + c*y) for x, y in base]

def collides(cx, cy, yaw, base):
    """检查：1) 膨胀后的单元格 2) 障碍格顶点 3) 出界"""
    pts = rotated_footprint(yaw, base)
    for dx, dy in pts:
        px, py = cx + dx, cy + dy
        gx, gy = int(round(px)), int(round(py))
        if gx < 0 or gx >= W or gy < 0 or gy >= H:
            return True
        if INFLATED_MAP[gx, gy]:
            return True
        cxn, cyn = int(round(px)), int(round(py))
        if cxn < 0 or cxn > W or cyn < 0 or cyn > H:
            return True
        if CORNER_BLOCK[cxn, cyn]:
            return True
    return False

# =========================== A* ==============================
hash3  = lambda i, j, k: (k << 20) | (i << 10) | j
octile = lambda dx, dy: math.sqrt(2)*min(dx, dy) + abs(dx - dy)
NB = [(dx, dy, dz) for dx in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                    for dz in (-1, 0, 1)
                    if not (dx == dy == dz == 0)]

def swept_safe_move(x0, y0, yaw0, x1, y1, yaw1, base):
    R = max(1e-6, max(math.hypot(x, y) for x, y in base))
    motion_dist = math.hypot(x1 - x0, y1 - y0)
    yaw_dist = abs(shortest_angle_diff(yaw0, yaw1))

    max_linear_step  = min(0.35, max(0.2, R/3.0))
    max_angular_step = math.radians(2)

    linear_steps  = max(1, int(math.ceil(motion_dist / max_linear_step)))
    angular_steps = max(1, int(math.ceil(yaw_dist / max_angular_step)))
    steps = max(linear_steps, angular_steps)

    for s in range(steps + 1):
        t = s / steps
        xi = x0 + (x1 - x0) * t
        yi = y0 + (y1 - y0) * t
        psi = (yaw0 + shortest_angle_diff(yaw0, yaw1) * t) % (2*math.pi)
        if collides(xi, yi, psi, base):
            return False
    return True

def backtrack(parent, last):
    path = [last]
    nid = hash3(int(last[0]), int(last[1]), yaw_bin(last[2]))
    while nid in parent:
        last = parent[nid]
        nid = hash3(int(last[0]), int(last[1]), yaw_bin(last[2]))
        path.append(last)
    return list(reversed(path))

def astar(start_xy, goal_xy, start_yaw, goal_yaw, base, max_pops=6000):
    """用于覆盖规划中“绕开障碍”的局部连接：在 (x,y,yaw) 网格上找一条安全路径。
    max_pops 限制搜索节点展开数量的上限，避免个别极难连通的目标把整体规划拖慢
    （超过预算就当作“暂时够不到”，规划器会换别的锚点/目标）。"""
    sx, sy = start_xy; gx, gy = goal_xy
    k0 = yaw_bin(start_yaw); kg = yaw_bin(goal_yaw)
    yaw0 = bin_to_yaw(k0); yg = bin_to_yaw(kg)

    if collides(sx, sy, yaw0, base) or collides(gx, gy, yg, base):
        return [], False

    sid = hash3(sx, sy, k0)
    g = {sid: 0.0}
    parent = {}
    open_q = [(octile(abs(gx-sx), abs(gy-sy)), sid, (sx, sy, yaw0))]
    closed = set()
    pops = 0

    while open_q:
        pops += 1
        if pops > max_pops:
            return [], False
        _, nid, (x, y, yaw) = heapq.heappop(open_q)
        kb = yaw_bin(yaw)
        if (int(x), int(y)) == (gx, gy) and kb == kg:
            return backtrack(parent, (x, y, yaw)), True
        if nid in closed:
            continue
        closed.add(nid)
        for dx, dy, dz in NB:
            nx, ny = x + dx, y + dy
            nk = (kb + dz) % YAW_DIV
            new_yaw = bin_to_yaw(nk)
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if not swept_safe_move(x, y, yaw, nx, ny, new_yaw, base):
                continue
            nid2 = hash3(int(nx), int(ny), nk)
            tg = g[nid] + math.hypot(dx, dy) + (abs(dz) * 0.3)
            if tg < g.get(nid2, 1e9):
                parent[nid2] = (x, y, yaw)
                g[nid2] = tg
                f2 = tg + octile(abs(gx - nx), abs(gy - ny))
                heapq.heappush(open_q, (f2, nid2, (nx, ny, new_yaw)))
    return [], False





if __name__=='__main__':
    main()
