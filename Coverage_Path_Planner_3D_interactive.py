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

# ----- 角点阻塞：把每个障碍格子的四个顶点也视为障碍 -----
CORNER_BLOCK = np.zeros((W + 1, H + 1), dtype=np.uint8)
obs_i, obs_j = np.where(MAP == 1)
for i, j in zip(obs_i, obs_j):
    CORNER_BLOCK[i,   j  ] = 1
    CORNER_BLOCK[i+1, j  ] = 1
    CORNER_BLOCK[i,   j+1] = 1
    CORNER_BLOCK[i+1, j+1] = 1

# ----- 欧式膨胀：保证与障碍的欧式距离 >= CLEARANCE_CELLS -----
def inflate_map_euclidean(M, margin):
    w, h = M.shape
    INFL = np.zeros_like(M, dtype=np.uint8)
    r = int(math.ceil(margin))
    maxd2 = margin * margin + 1e-9
    disk = [(dx, dy) for dx in range(-r, r + 1)
                      for dy in range(-r, r + 1)
                      if dx*dx + dy*dy <= maxd2]
    for dx, dy in disk:
        xs = slice(max(0, dx), min(w, w + dx))
        ys = slice(max(0, dy), min(h, h + dy))
        xsrc = slice(max(0, -dx), min(w, w - dx))
        ysrc = slice(max(0, -dy), min(h, h - dy))
        INFL[xs, ys] |= M[xsrc, ysrc]
    return INFL

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

# ============= 覆盖专用 2D A*（固定朝向，速度更快） =============
NB2 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == dy == 0)]

def swept_safe_move_2d(x0, y0, x1, y1, yaw, base):
    R = max(1e-6, max(math.hypot(x, y) for x, y in base))
    dist = math.hypot(x1 - x0, y1 - y0)
    step = min(0.5, max(0.25, R / 3.0))
    n = max(1, int(math.ceil(dist / step)))
    for s in range(n + 1):
        t = s / n
        if collides(x0 + (x1-x0)*t, y0 + (y1-y0)*t, yaw, base):
            return False
    return True

def astar2d(start_xy, goal_xy, yaw, base):
    """固定朝向下的 2D A*，只用于覆盖行/段之间的绕障连接，比 3D 版快得多。"""
    sx, sy = start_xy; gx, gy = goal_xy
    if collides(sx, sy, yaw, base) or collides(gx, gy, yaw, base):
        return [], False
    sid = (sx, sy)
    g = {sid: 0.0}
    parent = {}
    open_q = [(octile(abs(gx-sx), abs(gy-sy)), sid)]
    closed = set()
    while open_q:
        _, (x, y) = heapq.heappop(open_q)
        if (x, y) == (gx, gy):
            path = [(x, y)]
            n = (x, y)
            while n in parent:
                n = parent[n]
                path.append(n)
            path.reverse()
            return [(px, py, yaw) for (px, py) in path], True
        if (x, y) in closed:
            continue
        closed.add((x, y))
        for dx, dy in NB2:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if collides(nx, ny, yaw, base):
                continue
            if abs(dx) == 1 and abs(dy) == 1:
                # 禁止穿对角线夹缝（两侧都被占用时不能斜穿）
                if collides(x + dx, y, yaw, base) and collides(x, y + dy, yaw, base):
                    continue
            nid = (nx, ny)
            tg = g[(x, y)] + math.hypot(dx, dy)
            if tg < g.get(nid, 1e9):
                parent[nid] = (x, y)
                g[nid] = tg
                heapq.heappush(open_q, (tg + octile(abs(gx-nx), abs(gy-ny)), nid))
    return [], False

# ===================== 覆盖（Coverage）规划 =====================
def compute_bounds(yaw, base):
    fp = rotated_footprint(yaw, base)
    xs = [p[0] for p in fp]; ys = [p[1] for p in fp]
    return min(xs), max(xs), min(ys), max(ys)

def free_x_segments(y_c, yaw, x_lo, x_hi, base):
    """扫描一行内所有可行 x（整数格），返回连续自由区间列表 [(xa,xb), ...]，xa<=xb。"""
    free_xs = [x for x in range(x_lo, x_hi + 1) if not collides(x, y_c, yaw, base)]
    segs = []
    if not free_xs:
        return segs
    seg_start = prev = free_xs[0]
    for x in free_xs[1:]:
        if x == prev + 1:
            prev = x
        else:
            segs.append((seg_start, prev))
            seg_start = prev = x
    segs.append((seg_start, prev))
    return segs

def try_connect(p0, p1, base):
    """尝试从 p0 走到 p1（朝向固定不变）：先试直线，不行就用固定朝向的 2D A* 绕行。
    弓字形主扫描全程只用固定朝向，不需要转体，所以不需要（也不必付出代价去）调用
    较慢的 3D A*；真正需要转体钻缝的“补漏”阶段在 mop_up_coverage 里单独处理。
    两者都失败则返回 None（规划器会另起一段轨迹，而不是画一条穿墙的直线）。"""
    x0, y0, yaw0 = p0
    x1, y1, yaw1 = p1
    if swept_safe_move_2d(x0, y0, x1, y1, yaw0, base):
        return [p1]
    path, ok = astar2d((int(round(x0)), int(round(y0))),
                        (int(round(x1)), int(round(y1))),
                        yaw0, base)
    if ok and len(path) >= 2:
        return path[1:]
    return None

def plan_coverage_path(base, yaw_deg=COVER_YAW_DEG, overlap_ratio=ROW_OVERLAP):
    """弓字形全覆盖规划：固定朝向，逐行来回扫描，行间/障碍间用 A* 绕行连接。"""
    yaw = bin_to_yaw(yaw_bin(deg2rad(yaw_deg)))
    x_min, x_max, y_min, y_max = compute_bounds(yaw, base)

    x_lo = int(math.ceil(-x_min));      x_hi = int(math.floor((W - 1) - x_max))
    y_lo = int(math.ceil(-y_min));      y_hi = int(math.floor((H - 1) - y_max))
    if x_lo > x_hi or y_lo > y_hi:
        return [], yaw

    row_span = max(1.0, y_max - y_min)
    step_y = max(1, int(math.floor(row_span * (1.0 - overlap_ratio))))

    rows = list(range(y_lo, y_hi + 1, step_y))
    if rows[-1] != y_hi:
        rows.append(y_hi)          # 保证覆盖到最上面一行，不留边缘死角

    # strokes: 每个 stroke 内部相邻两点之间都保证是“安全可达”的（可以直接插值）。
    # 如果某段实在连不通（真正被地图分割成两块），就另起一个新的 stroke，
    # 中间不画线、不插值，避免穿墙。
    strokes = [[]]

    def cur():
        return strokes[-1]

    def goto(target):
        """从当前 stroke 末尾安全地走到 target；连不通则开新 stroke。"""
        if not cur():
            cur().append(target)
            return
        nxt = try_connect(cur()[-1], target, base)
        if nxt is None:
            strokes.append([target])
        else:
            cur().extend(nxt)
            if cur()[-1][:2] != target[:2]:
                cur().append(target)

    for ridx, y_c in enumerate(rows):
        segs = free_x_segments(y_c, yaw, x_lo, x_hi, base)
        if not segs:
            continue
        if ridx % 2 == 1:                      # 奇数行反向走（弓字形）
            segs = [(b, a) for (a, b) in reversed(segs)]
        for (xa, xb) in segs:
            entry = (float(xa), float(y_c), yaw)
            exitp = (float(xb), float(y_c), yaw)
            goto(entry)
            goto(exitp)

    strokes = [s for s in strokes if s]
    return strokes, yaw

def build_dense_path(waypoints, step=1.0):
    """把一条 stroke 内的稀疏路点按 step（格）线性细分，保证覆盖标记时不漏格。"""
    if not waypoints:
        return []
    dense = [waypoints[0]]
    for (x0, y0, yaw0), (x1, y1, yaw1) in zip(waypoints, waypoints[1:]):
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(dist / step)))
        for s in range(1, n + 1):
            t = s / n
            xi = x0 + (x1 - x0) * t
            yi = y0 + (y1 - y0) * t
            psi = (yaw0 + shortest_angle_diff(yaw0, yaw1) * t) % (2*math.pi)
            dense.append((xi, yi, psi))
    return dense

def build_dense_multistroke(strokes, step=1.0):
    """把多段 stroke 分别细分后拼接；在 stroke 之间插入一个 None 作为“抬笔”标记，
    动画/统计时据此跳过 stroke 之间的连线与插值（因为它们本来就没有物理路径连着）。"""
    dense = []
    for i, s in enumerate(strokes):
        if i > 0:
            dense.append(None)
        dense.extend(build_dense_path(s, step=step))
    return dense

def mark_covered(cov, cx, cy, yaw, base):
    pts = rotated_footprint(yaw, base)
    for dx, dy in pts:
        px, py = cx + dx, cy + dy
        gx, gy = int(round(px)), int(round(py))
        if 0 <= gx < W and 0 <= gy < H and MAP[gx, gy] == 0:
            cov[gx, gy] = 1

# =================== 覆盖补漏（Mop-up，允许转向） ===================
# 弓字形是“固定朝向”扫的，遇到孤立障碍物旁边的犄角旮旯，
# 单一朝向的 footprint 往往伸不进去。这里做第二阶段：
#   1) 找出所有还没覆盖到的自由格子，按连通域分组（一个个“坑”）；
#   2) 对每个坑，尝试机器人中心落在坑附近、16 个朝向中的某一个，
#      挑“碰撞安全 + 能盖住这个坑里最多格子”的姿态；
#   3) 用允许中途任意转体的 3D A*，从当前位置走过去（这一步如果直线走不通，
#      A* 会自动绕开孤立障碍物 —— 效果上就是“绕障碍物一圈再拐进去”）；
#   4) 同一个坑可能要挪几个不同朝向的姿态才能扫干净，重复 2~3 直到扫满或放弃。

def shape_radius(base):
    return max(1e-6, max(math.hypot(x, y) for x, y in base))

def find_uncovered_clusters(covered):
    """把 (MAP==0)&(covered==0) 的格子按 8-连通分组，返回 [[(x,y),...], ...]。"""
    free_uncov = (MAP == 0) & (covered == 0)
    visited = np.zeros_like(free_uncov)
    clusters = []
    for x in range(W):
        for y in range(H):
            if free_uncov[x, y] and not visited[x, y]:
                comp = []
                dq = deque([(x, y)]); visited[x, y] = True
                while dq:
                    cx, cy = dq.popleft()
                    comp.append((cx, cy))
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < W and 0 <= ny < H and free_uncov[nx, ny] and not visited[nx, ny]:
                                visited[nx, ny] = True
                                dq.append((nx, ny))
                clusters.append(comp)
    return clusters

def best_cover_pose(cluster_cells, base, r_shape, max_centers=6):
    """在 cluster 附近 + 16 个朝向里，找碰撞安全且覆盖该 cluster 格子数最多的姿态。"""
    cluster_set = set(cluster_cells)
    n = len(cluster_cells)
    # 代表点：均匀抽样 cluster 内的格子，避免大坑时枚举太多
    if n > max_centers:
        idxs = [int(round(i * (n - 1) / (max_centers - 1))) for i in range(max_centers)]
        reps = [cluster_cells[i] for i in idxs]
    else:
        reps = cluster_cells

    R = max(1, int(math.ceil(r_shape)))
    ring = [(dx, dy) for dx in range(-R, R + 1) for dy in range(-R, R + 1)
            if dx*dx + dy*dy <= R*R]

    best_pose = None
    best_score = 0
    tried = set()
    for (rx, ry) in reps:
        for dx, dy in ring:
            cx, cy = rx + dx, ry + dy
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            if (cx, cy) in tried:
                continue
            tried.add((cx, cy))
            for k in range(YAW_DIV):
                yaw = bin_to_yaw(k)
                if collides(cx, cy, yaw, base):
                    continue
                fp = rotated_footprint(yaw, base)
                score = 0
                for fx, fy in fp:
                    gx, gy = int(round(cx + fx)), int(round(cy + fy))
                    if (gx, gy) in cluster_set:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_pose = (float(cx), float(cy), yaw)
    return best_pose, best_score

def try_connect_general(p0, p1, base, astar3d_max_pops=1200):
    """允许起止朝向不同的连接，按代价从低到高依次尝试：
    1) 直线扫掠（起止同朝向，或距离很近时）；
    2) “原地转体到目标朝向 + 固定该朝向走 2D A*”——大多数情况下最快，
       物理上也合理（先转身、再朝一个方向走过去）；
    3) 实在不行，才退到允许沿途连续转体的完整 3D A*（限制搜索节点数，
       避免个别难连的目标拖慢整体规划）。
    找到路径则返回不含 p0 的中间/终点序列，否则返回 None。"""
    x0, y0, yaw0 = p0
    x1, y1, yaw1 = p1

    if swept_safe_move(x0, y0, yaw0, x1, y1, yaw1, base):
        return [p1]

    # 方案 2：原地转体 + 定朝向 2D A*
    ix0, iy0 = int(round(x0)), int(round(y0))
    ix1, iy1 = int(round(x1)), int(round(y1))
    if not collides(ix0, iy0, yaw1, base):
        path2d, ok2d = astar2d((ix0, iy0), (ix1, iy1), yaw1, base)
        if ok2d and len(path2d) >= 1:
            out = []
            if yaw_bin(yaw0) != yaw_bin(yaw1):
                out.append((float(ix0), float(iy0), yaw1))  # 原地转体
            out.extend(path2d[1:])                          # path2d[0] 就是起点，跳过
            if not out or out[-1] != p1:
                out.append(p1)
            return out

    # 方案 3：完整 3D A*（限预算）
    path3, ok3 = astar((ix0, iy0), (ix1, iy1), yaw0, yaw1, base, max_pops=astar3d_max_pops)
    if ok3 and len(path3) >= 2:
        return path3[1:]
    return None

# ============= “总轨迹最短”覆盖规划（替代弓字形） =============
# 思路（集合覆盖 + 旅行商，是覆盖路径规划里常见的做法）：
#   1) 候选姿态：在地图上按一定间隔取候选中心点，每个点尝试全部 16 个朝向，
#      保留碰撞安全的 (x,y,yaw)，并记录它“能盖住哪些自由格子”；
#   2) 贪心集合覆盖：每次挑“还能新盖住最多格子”的候选姿态，直到覆盖率达标——
#      这样选出的姿态数量远少于弓字形的密集路点，本身就为“短轨迹”打好基础；
#   3) 旅行商近似（最近邻构造 + 2-opt 局部优化）：给这些姿态排一个尽量短的
#      访问顺序，而不是像弓字形那样机械地按行走；
#   4) 按排好的顺序依次连接（直线 -> 定朝向2D A* -> 转体2D A* -> 3D A*），
#      连不通的地方另起一段（不画穿墙线）。
#   之后仍会跑一遍 mop_up_coverage 扫尾，把集合覆盖没顾上的边角格子补上。

def generate_cover_candidates(base, stride=None):
    """在网格上按 stride 取候选中心点 × 16 个朝向，返回碰撞安全、且确实盖住
    至少一个自由格子的候选：[(x, y, yaw, frozenset({(gx,gy),...})), ...]"""
    r_shape = shape_radius(base)
    if stride is None:
        stride = max(1, int(round(r_shape * 0.7)))
    candidates = []
    for cx in range(0, W, stride):
        for cy in range(0, H, stride):
            for k in range(YAW_DIV):
                yaw = bin_to_yaw(k)
                if collides(cx, cy, yaw, base):
                    continue
                fp = rotated_footprint(yaw, base)
                cells = set()
                for dx, dy in fp:
                    gx, gy = int(round(cx + dx)), int(round(cy + dy))
                    if 0 <= gx < W and 0 <= gy < H and MAP[gx, gy] == 0:
                        cells.add(gx * H + gy)   # 压成一维索引，方便向量化打分
                if cells:
                    candidates.append((float(cx), float(cy), yaw, cells))
    return candidates

def greedy_set_cover(candidates, target_frac=0.97, max_iters=3000):
    """贪心集合覆盖：每步选“新增覆盖格子数最多”的候选，直到覆盖率达标或选不出更多。"""
    free_mask = (MAP == 0)
    free_total = int(free_mask.sum())
    remaining = np.zeros(W * H, dtype=bool)
    remaining[np.flatnonzero(free_mask.reshape(-1))] = True

    cell_arrays = [np.fromiter(c, dtype=np.int64, count=len(c)) for (_, _, _, c) in candidates]
    selected = []
    covered_count = 0
    it = 0
    n = len(candidates)
    alive = np.ones(n, dtype=bool)

    while covered_count < free_total * target_frac and it < max_iters:
        it += 1
        best_idx = -1
        best_score = 0
        for i in range(n):
            if not alive[i]:
                continue
            score = int(remaining[cell_arrays[i]].sum())
            if score == 0:
                alive[i] = False
                continue
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx == -1:
            break
        idxs = cell_arrays[best_idx]
        newly = remaining[idxs]
        covered_count += int(newly.sum())
        remaining[idxs] = False
        alive[best_idx] = False
        x, y, yaw, _ = candidates[best_idx]
        selected.append((x, y, yaw))

    return selected, covered_count / max(1, free_total)

def order_poses_shortest(poses, time_budget=6.0):
    """最近邻构造初始巡回顺序，再用 2-opt 局部优化缩短总长度（近似 TSP）。"""
    n = len(poses)
    if n <= 2:
        return list(poses)
    xs = np.array([p[0] for p in poses])
    ys = np.array([p[1] for p in poses])

    def dist(i, j):
        return math.hypot(xs[i] - xs[j], ys[i] - ys[j])

    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = order[-1]
        dx = xs - xs[last]; dy = ys - ys[last]
        d2 = dx*dx + dy*dy
        d2[visited] = np.inf
        j = int(np.argmin(d2))
        order.append(j)
        visited[j] = True

    import time as _time
    t0 = _time.time()
    improved = True
    while improved and _time.time() - t0 < time_budget:
        improved = False
        for i in range(1, n - 2):
            a, b = order[i - 1], order[i]
            dab = dist(a, b)
            for j in range(i + 1, n - 1):
                c, d = order[j], order[j + 1]
                if dab + dist(c, d) > dist(a, c) + dist(b, d) + 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
                    a, b = order[i - 1], order[i]
                    dab = dist(a, b)
        if _time.time() - t0 > time_budget:
            break
    return [poses[i] for i in order]

def build_path_from_poses(poses, base):
    """按顺序把姿态连起来；连不通的地方另起一段（不画穿墙线）。"""
    strokes = [[]]
    for p in poses:
        if not strokes[-1]:
            strokes[-1].append(p)
            continue
        nxt = try_connect_general(strokes[-1][-1], p, base)
        if nxt is None:
            strokes.append([p])
        else:
            strokes[-1].extend(nxt)
            if strokes[-1][-1] != p:
                strokes[-1].append(p)
    return [s for s in strokes if s]

def plan_coverage_path_shortest(base, target_frac=0.97):
    """“总轨迹最短”覆盖规划入口：集合覆盖选点 + 近似 TSP 排序 + 逐段连接。"""
    candidates = generate_cover_candidates(base)
    selected, frac = greedy_set_cover(candidates, target_frac=target_frac)
    if not selected:
        return [], bin_to_yaw(yaw_bin(deg2rad(COVER_YAW_DEG)))
    ordered = order_poses_shortest(selected)
    strokes = build_path_from_poses(ordered, base)
    print(f"   (集合覆盖选出 {len(selected)} 个姿态，覆盖自由格 {frac*100:.1f}%，"
          f"TSP 排序后拼成 {len(strokes)} 段轨迹)")
    return strokes, None

def mop_up_coverage(strokes, covered, base, dense_main,
                     max_per_cluster=5, max_total_moves=400,
                     max_anchors=250, anchors_tried_per_cluster=8):
    """在主弓字形之后追加若干“补漏”stroke。就地修改 covered（标记新覆盖格）。
    返回新增的 stroke 列表（可能为空）。

    关键点：不是只从“主路径最后一个点”出发去够每个坑（那一点很可能正好卡在
    死胡同里，导致到处都连不通）。而是把主路径上采样出的一批锚点都当作可能
    的出发点，对每个坑，从离它最近的若干个锚点里找一个真正连得通的，
    从那里“绕出去”覆盖这个坑、再回到（下一次连接时会自动重新找锚点）。
    这样效果上就是：弓字形扫到障碍物附近时，就近甩出一段绕障碍物的小尾巴，
    把犄角旮旯扫掉，再继续/或从别处接着扫。"""
    r_shape = shape_radius(base)

    # 采样锚点：主路径上的一批 (x,y,yaw)，作为潜在的“出发/返回”位置
    real_pts = [p for p in dense_main if p is not None]
    if len(real_pts) > max_anchors:
        step = max(1, len(real_pts) // max_anchors)
        anchors = real_pts[::step]
    else:
        anchors = list(real_pts)
    if not anchors:
        anchors = [strokes[-1][-1]] if strokes and strokes[-1] else [(0.0, 0.0, 0.0)]

    clusters = find_uncovered_clusters(covered)
    clusters = [c for c in clusters if len(c) >= 1]

    def cluster_centroid(c):
        xs = [p[0] for p in c]; ys = [p[1] for p in c]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    # 先按“离主路径最近”排序处理（离得近的坑更容易顺路捎带覆盖，动画也更自然）
    def dist_cluster_to_anchors(c):
        ccx, ccy = cluster_centroid(c)
        return min(math.hypot(ax - ccx, ay - ccy) for (ax, ay, _) in anchors)

    clusters.sort(key=dist_cluster_to_anchors)

    new_strokes = []
    moves = 0
    for cluster in clusters:
        if moves >= max_total_moves:
            break
        ccx, ccy = cluster_centroid(cluster)
        ranked_anchors = sorted(anchors, key=lambda a: math.hypot(a[0]-ccx, a[1]-ccy))
        anchor_pool = ranked_anchors[:anchors_tried_per_cluster]

        cur_stroke = []
        last_pose = None
        attempts = 0
        while attempts < max_per_cluster and moves < max_total_moves:
            remaining = [(x, y) for (x, y) in cluster if covered[x, y] == 0]
            if not remaining:
                break
            pose, score = best_cover_pose(remaining, base, r_shape)
            if pose is None or score == 0:
                break

            nxt = None
            origin = None
            if last_pose is not None:
                # 已经在坑边上了，优先原地继续（大概率直接连通，省一次搜索）
                nxt = try_connect_general(last_pose, pose, base)
                origin = last_pose
            if nxt is None:
                for a in anchor_pool:
                    cand = try_connect_general(a, pose, base)
                    if cand is not None:
                        nxt, origin = cand, a
                        break
            attempts += 1
            if nxt is None:
                # 这个坑目前谁都够不到（大概率被完全围死），放弃它，换下一个坑
                break

            if last_pose is None or origin != last_pose:
                # 从一个新的锚点出发：另起一段 stroke（动画上表现为“甩出一段小尾巴”）
                if cur_stroke:
                    new_strokes.append(cur_stroke)
                cur_stroke = [origin]
            cur_stroke.extend(nxt)
            last_pose = pose
            mark_covered(covered, pose[0], pose[1], pose[2], base)
            moves += 1
        if cur_stroke:
            new_strokes.append(cur_stroke)
    return new_strokes


# =================== 交互显示层（不参与规划） ===================
# python Coverage_Path_Planner_3D_interactive.py --shape F
# GUI: 空格暂停；右箭头单步；R复位视角；S保存300dpi截图。
# 鼠标左键拖动旋转、滚轮缩放；可用 Matplotlib 工具栏平移。
# 无窗口导出: python Coverage_Path_Planner_3D_interactive.py --shape F --headless
# 注意：沿用原算法的离散点碰撞/覆盖模型；实心轮廓是原ASCII单元的刚性外观，
# 并不将原算法升级成连续多边形碰撞检测。统计仅由原 dense_path 计算。
import argparse
import time
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PAPER = '#f7f5ef'
INK = '#283c45'
TEAL = '#159a9b'

def body_geometry(base):
    """共享边消除：只画真正外轮廓，保留孔洞和凹口。"""
    cells, edges = [], {}
    for x, y in base:
        q = [(x-.5,y-.5),(x+.5,y-.5),(x+.5,y+.5),(x-.5,y+.5)]
        cells.append(q)
        for a,b in zip(q,q[1:]+q[:1]):
            key=tuple(sorted((a,b)))
            if key in edges:
                del edges[key]
            else:
                edges[key]=(a,b)
    return np.asarray(cells,float), np.asarray(list(edges.values()),float)


def transform_body(points, pose):
    x,y,yaw=pose
    c,s=math.cos(yaw),math.sin(yaw)
    return points @ np.array([[c,s],[-s,c]]) + [x,y]


def xyz(points,z):
    return np.concatenate([points,np.full((*points.shape[:-1],1),z)],axis=-1)


def box_faces(x,y,h):
    # MAP索引是round(x), round(y)：栅格以整数坐标为中心。
    a,b=x-.5,y-.5
    v=np.array([[a,b,0],[a+1,b,0],[a+1,b+1,0],[a,b+1,0],
                [a,b,h],[a+1,b,h],[a+1,b+1,h],[a,b+1,h]])
    return v[np.array([[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]])]


class CoveragePlayer:
    """保留图元与相机；按原采样点推进，支持暂停、拖动进度、回放。"""
    def __init__(self,path,base,shape_name,output_dir):
        self.path=list(path)
        self.base=base
        self.shape_name=shape_name
        self.output_dir=Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.cells,self.edges=body_geometry(base)
        self.covered=np.zeros((W,H),np.uint8)
        self.index=0
        self.pose=None
        self.tx,self.ty=[],[]
        self.bad=0
        self.break_pending=False
        self.playing=False
        self.dragging=False
        self.credit=0.
        self.last_time=time.monotonic()
        self.syncing=False
        self.free=max(1,int((MAP==0).sum()))
        self.fig=plt.figure(figsize=(14,8.5),dpi=110,facecolor=PAPER)
        self.ax=self.fig.add_axes([.015,.19,.715,.75],projection='3d',computed_zorder=False)
        self.ax.set_facecolor(PAPER)
        self.ax.set_proj_type('persp',focal_length=1.15)
        self.ax.set_box_aspect((W,H,9))
        self.ax.set_axis_off()
        self.reset_camera()
        self.fig.text(.05,.95,'SHAPE-AWARE COVERAGE',color=INK,fontsize=18,weight='bold')
        self.fig.text(.05,.922,f'{shape_name} footprint  /  original planner  /  interactive replay',
                      color='#647078',fontsize=10)
        self.status=self.fig.text(.05,.18,'',fontsize=11,color=INK)
        self.fig.text(.05,.02,'Drag: orbit    Wheel: zoom    Space: play/pause    Right: step    R: reset view    S: PNG',
                      fontsize=9,color='#647078')
        self.local=self.fig.add_axes([.75,.49,.225,.36],facecolor=PAPER)
        self.local.set_aspect('equal')
        self.local.set_title('LIVE FOOTPRINT · TOP VIEW',fontsize=10,color=INK,pad=12)
        self.local.tick_params(labelsize=8,colors='#7a8589')
        for spine in self.local.spines.values(): spine.set_color('#b9c1c2')
        self.local.set_xlabel('Map coordinates · same scale in x/y',fontsize=8,color='#647078')
        # floor tiles retain exact coverage boundaries; no blur on obstacles.
        gridcells=np.array([[(x-.5,y-.5),(x+.5,y-.5),(x+.5,y+.5),(x-.5,y+.5)]
                            for x in range(W) for y in range(H)])
        self.floor=Poly3DCollection(xyz(gridcells,0),linewidths=0,antialiaseds=False,zorder=1)
        self.ax.add_collection3d(self.floor)
        self.local_map=self.local.imshow(np.zeros((H,W,3)),origin='lower',
                         extent=(-.5,W-.5,-.5,H-.5),interpolation='nearest')
        # Crisp opaque obstacles. Full unit-cell footprint matches the displayed map.
        obstacle_faces=[]
        obscolors=[]
        for x,y in zip(*np.where(MAP==1)):
            h=2.2+1.1*(.5+.5*math.sin(.41*x+.67*y+.8))
            obstacle_faces.extend(box_faces(x,y,h))
            obscolors.extend(['#f4f5f1','#b4c0c5','#7f939e','#a3b2ba','#dce2e1'])
        self.obstacles=Poly3DCollection(obstacle_faces,facecolors=obscolors,
                            edgecolors='#536975',linewidths=.65,antialiaseds=True,zorder=10)
        self.ax.add_collection3d(self.obstacles)
        self.ax.plot([-.5,W-.5,W-.5,-.5,-.5],[-.5,-.5,H-.5,H-.5,-.5],
                     [0]*5,color='#809097',lw=.8)
        # Exact centre path: rounded stroke styling without changing coordinates.
        self.glow,=self.ax.plot([],[],[],color='#aa7250',lw=4,alpha=.13,solid_capstyle='round',zorder=2)
        self.trail,=self.ax.plot([],[],[],color='#875536',lw=1.2,alpha=.85,solid_capstyle='round',zorder=3)
        self.body=Poly3DCollection([],facecolors=TEAL,edgecolors='none',antialiaseds=False,zorder=4)
        self.outline=Line3DCollection(np.zeros((1,2,3)),colors='#073f49',linewidths=1.7,zorder=5)
        self.ax.add_collection3d(self.body)
        self.ax.add_collection3d(self.outline)
        self.center,=self.ax.plot([],[],[],marker='o',markersize=3.8,color='#d44536',linestyle='None',zorder=6)
        self.local_body=PolyCollection([],facecolors=TEAL,edgecolors='none',antialiaseds=False,zorder=4)
        self.local_outline=LineCollection([],colors='#073f49',linewidths=2,zorder=5)
        self.local.add_collection(self.local_body)
        self.local.add_collection(self.local_outline)
        self.local_center,=self.local.plot([],[],'o',color='#d44536',ms=4,zorder=6)
        self.local_trail,=self.local.plot([],[],color='#875536',lw=1.2,zorder=3)
        self.detail=self.fig.text(.75,.40,'',fontsize=9,color=INK,linespacing=1.6,va='top')
        self.fig.text(.75,.265,'■  Covered cells: apricot\n■  Current footprint: teal\n■  Obstacles: slate / ivory',
                      fontsize=10,color=INK,linespacing=1.7)
        self.buttons=[]
        for label,left,width,callback in [
                ('Play',.05,.085,self.toggle),('Step',.145,.07,self.step),
                ('Reset view',.225,.095,self.reset_camera),('Top view',.33,.085,self.top_view),
                ('Save PNG',.425,.085,self.snapshot)]:
            b=Button(self.fig.add_axes([left,.105,width,.038]),label,color='#e8ede9',hovercolor='#d3e4df')
            b.on_clicked(callback)
            self.buttons.append(b)
        self.speed=Slider(self.fig.add_axes([.65,.11,.27,.021]),'Samples/s',1,40,valinit=6,valstep=1)
        self.progress=Slider(self.fig.add_axes([.10,.061,.80,.019]),'Progress',0,100,valinit=0,valfmt='%1.1f%%')
        self.progress.on_changed(self.seek_percent)
        self.timer=self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect('key_press_event',self.key)
        self.fig.canvas.mpl_connect('scroll_event',self.zoom)
        self.fig.canvas.mpl_connect('button_press_event',self.press)
        self.fig.canvas.mpl_connect('button_release_event',self.release)
        self.fig.canvas.mpl_connect('close_event',lambda event:self.timer.stop())
        self.render()

    def floor_colors(self):
        cov=(self.covered>0)&(MAP==0)
        # A small halo is decorative only; solid tiles are the measured coverage.
        halo=np.zeros((W,H),float)
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            rolled=np.roll(cov,(dx,dy),axis=(0,1)).astype(float)
            if dx==1: rolled[0,:]=0
            if dx==-1: rolled[-1,:]=0
            if dy==1: rolled[:,0]=0
            if dy==-1: rolled[:,-1]=0
            halo+=rolled
        alpha=.60*cov+.018*halo*(~cov)
        rgb=np.ones((W,H,3))*np.array([.956,.953,.928])
        rgb=rgb*(1-alpha[...,None])+np.array([.89,.57,.32])*alpha[...,None]
        rgb[MAP==1]=[.34,.43,.48]
        return rgb

    def reset_camera(self,event=None):
        self.ax.set_xlim(-1,W)
        self.ax.set_ylim(-1,H)
        self.ax.set_zlim(0,4)
        self.ax.view_init(elev=52,azim=-58)
        self.fig.canvas.draw_idle()

    def top_view(self,event=None):
        self.ax.view_init(elev=90,azim=-90)
        self.fig.canvas.draw_idle()

    def zoom(self,event):
        if event.inaxes!=self.ax: return
        factor=.88 if event.button=='up' else 1/.88
        for get,setter in [(self.ax.get_xlim,self.ax.set_xlim),(self.ax.get_ylim,self.ax.set_ylim)]:
            lo,hi=get(); mid=(lo+hi)/2; span=np.clip((hi-lo)*factor,6,160)/2
            setter(mid-span,mid+span)
        self.fig.canvas.draw_idle()

    def press(self,event):
        if event.inaxes in (self.ax,self.progress.ax): self.dragging=True

    def release(self,event):
        self.dragging=False
        self.last_time=time.monotonic()

    def consume(self,end):
        for p in self.path[self.index:end]:
            if p is None:
                self.tx.append(np.nan); self.ty.append(np.nan)
                self.pose=None
                self.break_pending=True
                continue
            if collides(*p,self.base):
                self.bad+=1
                self.tx.append(np.nan);self.ty.append(np.nan)
                self.pose=None
                continue
            mark_covered(self.covered,*p,self.base)
            self.pose=p
            self.break_pending=False
            self.tx.append(p[0]);self.ty.append(p[1])
        self.index=end

    def seek(self,end):
        end=int(np.clip(end,0,len(self.path)))
        if end<self.index:
            self.covered.fill(0)
            self.tx,self.ty=[],[]
            self.pose=None
            self.index=0;self.bad=0;self.break_pending=False
        self.consume(end)
        self.render()

    def seek_percent(self,value):
        if self.syncing:return
        self.seek(round(value*len(self.path)/100))
        self.credit=0
        self.last_time=time.monotonic()

    def render(self):
        rgb=self.floor_colors()
        self.floor.set_facecolors(rgb.reshape(-1,3))
        self.local_map.set_data(rgb.transpose(1,0,2))
        self.glow.set_data_3d(self.tx,self.ty,np.full(len(self.tx),.025))
        self.trail.set_data_3d(self.tx,self.ty,np.full(len(self.tx),.035))
        self.local_trail.set_data(self.tx,self.ty)
        if self.pose is not None:
            cells=transform_body(self.cells,self.pose)
            edges=transform_body(self.edges,self.pose)
            self.body.set_verts(xyz(cells,.075))
            self.outline.set_segments(xyz(edges,.085))
            self.local_body.set_verts(cells)
            self.local_outline.set_segments(edges)
            x,y,yaw=self.pose
            self.center.set_data_3d([x],[y],[.095])
            self.local_center.set_data([x],[y])
            radius=max(4.5,float(np.linalg.norm(self.cells,axis=-1).max())+1.8)
            self.local.set_xlim(x-radius,x+radius)
            self.local.set_ylim(y-radius,y+radius)
            self.detail.set_text(f'Pose  ({x:.2f}, {y:.2f})\nYaw   {math.degrees(yaw)%360:.1f}°\nRigid footprint · no scale change')
        else:
            self.body.set_verts([]);self.outline.set_segments([])
            self.local_body.set_verts([]);self.local_outline.set_segments([])
            self.center.set_data_3d([],[],[]);self.local_center.set_data([],[])
            if not self.index:
                self.local.set_xlim(-.5,W-.5);self.local.set_ylim(-.5,H-.5)
            self.detail.set_text('Stroke break · no connecting motion' if self.break_pending else 'Ready / no valid pose')
        pct=100*int(((self.covered>0)&(MAP==0)).sum())/self.free
        state='Complete' if self.index==len(self.path) else ('Playing' if self.playing else 'Paused')
        self.status.set_text(f'{state}    Coverage {pct:.2f}%    Sample {self.index:,}/{len(self.path):,}    Unsafe samples skipped: {self.bad}')
        self.syncing=True
        self.progress.set_val(100*self.index/max(1,len(self.path)))
        self.syncing=False
        self.fig.canvas.draw_idle()

    def toggle(self,event=None):
        if self.index==len(self.path) and not self.playing: self.seek(0)
        self.playing=not self.playing
        self.buttons[0].label.set_text('Pause' if self.playing else 'Play')
        self.last_time=time.monotonic(); self.credit=0
        self.render()

    def step(self,event=None):
        self.playing=False
        self.buttons[0].label.set_text('Play')
        self.seek(self.index+1)

    def tick(self):
        now=time.monotonic()
        elapsed=min(now-self.last_time,.20)
        self.last_time=now
        if not self.playing or self.dragging:return True
        self.credit+=elapsed*self.speed.val
        n=int(self.credit)
        if n:
            self.credit-=n
            if self.index+n>=len(self.path):
                self.playing=False; self.buttons[0].label.set_text('Play')
            self.seek(min(self.index+n,len(self.path)))
        return True

    def key(self,event):
        if event.key==' ':self.toggle()
        elif event.key=='right':self.step()
        elif event.key=='r':self.reset_camera()
        elif event.key=='s':self.snapshot()

    def snapshot(self,event=None):
        path=self.output_dir/f'coverage_interactive_{self.shape_name}_{self.index:06d}.png'
        self.fig.savefig(path,dpi=300,facecolor=PAPER)
        print(f'PNG: {path.resolve()}',flush=True)
        return path

    def show(self):
        # Event-loop driven replay. Planning is completed before this window opens.
        self.timer.start()
        self.toggle()
        plt.show()


def main():
    parser=argparse.ArgumentParser(description='Original coverage planner with interactive 3D replay')
    parser.add_argument('--shape',choices=['F','T','A','all'],default='F')
    parser.add_argument('--headless',action='store_true',help='Export final PNG without a GUI')
    parser.add_argument('--output',default='coverage_output')
    args=parser.parse_args()
    if not args.headless and str(matplotlib.get_backend()).lower()=='agg':
        raise SystemExit('No GUI backend is available. Run in a desktop Python with Tk/Qt, or use --headless for PNG export.')
    output_dir=Path(args.output)
    output_dir.mkdir(parents=True,exist_ok=True)
    shapes=SHAPES_TO_RUN if args.shape=='all' else [args.shape]
    for shape_name in shapes:
        base=SHAPES[shape_name]
        print(f'[{shape_name}] Planning with the original algorithm; interactive replay opens after planning.',flush=True)
        if COVERAGE_STRATEGY == "shortest":
            strokes, cov_yaw = plan_coverage_path_shortest(base)
        else:
            strokes, cov_yaw = plan_coverage_path(base)
        if not strokes:
            print(f"❌ [{shape_name}] 机器人相对于地图太大，无法规划覆盖路径")
            continue

        # ---- 主覆盖阶段（弓字形 或 最短轨迹集合覆盖+TSP，取决于 COVERAGE_STRATEGY）----
        dense_main = build_dense_multistroke(strokes, step=0.5)
        COVERED = np.zeros((W, H), dtype=np.uint8)
        for p in dense_main:
            if p is None:
                continue
            x, y, yaw = p
            if not collides(x, y, yaw, base):
                mark_covered(COVERED, x, y, yaw, base)
        main_pct = 100.0 * int(((MAP == 0) & (COVERED == 1)).sum()) / max(1, int((MAP == 0).sum()))

        # ---- 补漏阶段：允许转向，专门去啃边角/孤立障碍物旁边的坑 ----
        mopup_strokes = mop_up_coverage(strokes, COVERED, base, dense_main)
        strokes = strokes + mopup_strokes
        print(f"   [{shape_name}] 主覆盖阶段（{COVERAGE_STRATEGY}） {main_pct:.1f}%，补漏追加 {len(mopup_strokes)} 段轨迹")

        dense_path = build_dense_multistroke(strokes, step=0.5)
        n_strokes = len(strokes)
        n_waypoints = sum(len(s) for s in strokes)

        # ---- 再离线跑一遍完整路径，统计总覆盖率，并检查安全性 ----
        COVERED = np.zeros((W, H), dtype=np.uint8)
        unsafe_count = 0
        n_real = 0
        for p in dense_path:
            if p is None:
                continue
            n_real += 1
            x, y, yaw = p
            if collides(x, y, yaw, base):
                unsafe_count += 1
                continue
            mark_covered(COVERED, x, y, yaw, base)
        free_cells = int((MAP == 0).sum())
        covered_cells = int(((MAP == 0) & (COVERED == 1)).sum())
        coverage_pct = 100.0 * covered_cells / max(1, free_cells)
        traj_len = 0.0
        prev = None
        for p in dense_path:
            if p is None:
                prev = None
                continue
            if prev is not None:
                traj_len += math.hypot(p[0]-prev[0], p[1]-prev[1])
            prev = p
        print(f"✔️ [{shape_name}] stroke段数={n_strokes}  路点数={n_waypoints}  细分点数={n_real}  "
              f"总轨迹长度≈{traj_len:.0f} 格  覆盖率={coverage_pct:.1f}%  (不安全采样点={unsafe_count})")

        player=CoveragePlayer(dense_path,base,shape_name,output_dir)
        if args.headless:
            player.seek(len(dense_path))
            if not np.array_equal(player.covered,COVERED):
                raise RuntimeError('Replay coverage differs from original planner statistics')
            player.snapshot()
            plt.close(player.fig)
        else:
            player.show()

if __name__=='__main__':
    main()
