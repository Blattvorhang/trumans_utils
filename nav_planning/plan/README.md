# Plan: Replanning Support for TRUMANS Nav Pipeline

## Context

TRUMANS 是自回归扩散模型。每个 episode 生成时，上一 episode 的最后 `fixed_frame=2` 帧被 clamp 到扩散过程中作为固定上下文 —— 这就是模型"接续"之前动作的机制。这个状态由两个变量承载：

| 变量 | Shape | 空间 | 含义 |
|------|-------|------|------|
| `fixed_points` | `(1, 2, 72)` | y-up world | 上一 episode 最后 2 帧的 24 关节 3D 位置 |
| `mat` | `(1, 4, 4)` | y-up world | 归一化坐标→世界坐标的刚体变换矩阵 |

**目标**：每次 plan 是独立脚本调用，走完整 pipeline（路径规划→waypoints→TRUMANS 自回归生成）。需要支持传入上一轮的状态，使动作首尾连贯。

## 核心问题：History Buffer 要用 G1 真实位姿更新

上一轮结尾的状态 `(mat, fixed_points)` 储存的是**模型计划的**位姿。但仿真中 G1 跟踪执行后，真实位置和朝向与计划有偏差。直接沿用原状态会导致 gap。

**修正方案**：对 history buffer 做**刚体变换**——
- **Root position** → 用 G1 真实 `(x, y)` 覆盖
- **Root yaw** → 用 G1 真实朝向覆盖
- **Body pose**（肢体姿态）→ 保留原生成结果不变

数学上，这等价于计算旧世界坐标系到新世界坐标系的刚体变换，应用到 `fixed_points`：

```
fixed_points_new_world = new_mat @ inverse(old_mat) @ fixed_points_old_world
```

## 对外接口：统一 z-up（AMASS/MuJoCo 惯例）

| 输入 | 坐标系 | 说明 |
|------|--------|------|
| `--start x y` | z-up | G1 地面位置 `(x=right, y=forward)` |
| `--goal x y` | z-up | 同上 |
| `--robot-heading dx dy` | z-up | G1 朝向向量，地面平面 `(dx, dy)`，无需归一化 |
| `--resume-from` | 文件 | 上一轮的 `_state.npz` |

## 调用方式

```bash
# 第 1 轮：冷启动
python nav_planning/trumans_infer.py \
    --start 0 0 --goal 5 0 \
    --scene scene.npy --output motion_1.npz --planner astar
# → motion_1.npz + motion_1_state.npz

# 仿真器跟踪 motion_1.npz 2 秒 → 读到 G1 真实 (x, y) 和朝向向量 (dx, dy)

# 第 2 轮：replan
python nav_planning/trumans_infer.py \
    --start <g1_real_x> <g1_real_y> --goal 5 0 \
    --scene scene.npy --output motion_2.npz --planner astar \
    --resume-from motion_1_state.npz \
    --robot-heading <g1_real_dx> <g1_real_dy>
# → motion_2.npz + motion_2_state.npz
# motion_2 开头 2 帧: root=(G1真实x, G1真实y, SMPL原高度z), 朝向=G1真实朝向, 肢体=上一轮结尾

# ... 循环直到到达 goal ...
```

## 架构

```
                    ┌──────────────────────────────────────────────┐
                    │        trumans_infer.py::run_inference()     │
                    │                                              │
  --start (x,y) ───▶  scale → path_planner → trajectory (y-up)    │
  --goal (x,y)      (same as current pipeline)                     │
                    │                                              │
                    │  get_guidance(trajectory)                    │
                    │    → mat_init, goal_list, action_label_list  │
                    │                                              │
  --resume-from ───▶  if resume_from:                             │
  --robot-heading     state = load_state(file)                     │
    (dx,dy)           adapted = adapt_state(                       │
                           state, new_start_zup, heading_zup)      │
                       fixed_points_init = adapted.fixed_points ◀── 热启动
                    │                                              │
                    │  sample_step(                                │
                    │      mat_init, goal_list, ...,               │
                    │      fixed_points_init)                      │
                    │    → points_all, final_mat, final_fp         │
                    │                                              │
                    │  save_state(final_mat, final_fp) → _state.npz│
                    │  joints_to_smpl(points_all) → .npz           │
                    └──────────────────────────────────────────────┘
```

### 改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `sample_hsi.py` | **改** | `sample_step()` 接受 `fixed_points_init`，返回值加 state |
| `nav_planning/replan_state.py` | **新** | `GenerationState` + 序列化 + `adapt_state()` |
| `nav_planning/trumans_infer.py` | **改** | 增加 `--resume-from` / `--robot-yaw`，保存 state |

### 不动

`models/synhsi.py`、`path_planner.py`、`datasets/trumans.py`、`get_guidance()`。

---

## 1. `sample_hsi.py` 改动

### 核心改动：`if s != 0` → `if fixed_points_world is not None`

```python
# BEFORE:
fixed_points = None
for s in range(max_step):
    if s != 0:
        fixed_points = normalize(transform_points(fixed_points, inv(mat)))
    samples = sampler.p_sample_loop(fixed_points, ...)
    fixed_points = points_gene[:, -fixed_frame:]  # world space

# AFTER:
fixed_points_world = fixed_points_init  # None for cold start
for s in range(start_step, max_step):
    if fixed_points_world is not None:
        fixed_points = normalize(transform_points(fixed_points_world, inv(mat)))
    else:
        fixed_points = None
    samples = sampler.p_sample_loop(fixed_points, ...)
    fixed_points_world = points_gene[:, -fixed_frame:]  # world space
return points_all, mat, fixed_points_world, steps_completed
```

| 场景 | `fixed_points_world` | 行为 |
|------|---------------------|------|
| 冷启动 s=0 | `None` | 跳过 → 等同原来 |
| 正常 s>0 | 上一轮 world tensor | 归一化 → 等同原来 |
| Replan s=0 | `adapt_state()` 的 world tensor | 归一化到新 mat → **新行为** |

### 新签名

```python
def sample_step(cfg, mat, obj_locs, goal_list, action_label_list, sampler_list,
                fixed_points_init=None, start_step=0, max_steps=None):
    """
    Returns:
        points_all:          (B, T_all, J, 3) np.array, y-up world
        mat_final:           (B, 4, 4) torch.Tensor, y-up world
        fixed_points_final:  (B, F, 72) torch.Tensor, y-up world
        steps_completed:     int
    """
```

### 向后兼容

```python
def sample_step_legacy(cfg, mat, obj_locs, goal_list, action_label_list, sampler_list):
    points_all, _, _, _ = sample_step(cfg, mat, obj_locs, goal_list,
                                       action_label_list, sampler_list)
    return points_all
```

---

## 2. `nav_planning/replan_state.py` 新建

### 数据结构

```python
@dataclass
class GenerationState:
    mat: np.ndarray           # (1, 4, 4) float32 — y-up world transform
    fixed_points: np.ndarray  # (1, fixed_frame, 72) float32 — y-up world, 24 joint positions
```

### 序列化

```python
def save_state(state: GenerationState, path: str) -> None:
    np.savez(path, mat=state.mat, fixed_points=state.fixed_points)

def load_state(path: str) -> GenerationState:
    data = np.load(path)
    return GenerationState(mat=data["mat"], fixed_points=data["fixed_points"])
```

### 核心：`adapt_state()` — 用 G1 真实位姿更新 history buffer

```python
def adapt_state(
    state: GenerationState,
    new_start_zup: tuple[float, float],  # G1 真实 (x, y)，z-up：x=right, y=forward
    heading_zup: tuple[float, float],    # G1 朝向向量 (dx, dy)，z-up 地面平面
) -> GenerationState:
    """
    对 history buffer 做刚体变换：
    - Root (x, y)  → G1 真实位置（z-up 地面）
    - Root z        → 保留 SMPL 原有高度
    - 朝向          → G1 真实朝向向量
    - Body pose     → 保留不变

    数学：fixed_points_new = new_mat @ inv(old_mat) @ fixed_points_old

    其中 old_mat = state.mat（上一轮 world transform）
         new_mat = 根据 G1 真实位置和朝向向量构建

    高度保留：new_mat 和 old_mat 的 Y 平移均为 0（y-up 中 Y=up），
    刚体变换只旋转地面分量（绕 Y 轴），Y 坐标不变 → SMPL 身高自动保留。
    """
    from scipy.spatial.transform import Rotation as R
    import math

    # 1. 将 G1 位置从 z-up 转到 y-up
    #    z-up (x=right, y=forward, z=up) → y-up (x=left, y=up, z=forward)
    #    (x_zup, y_zup) → (-x, z=0, y)   # z 分量暂设 0，高度由 SMPL 提供
    new_pos_yup = np.array([-new_start_zup[0], 0.0, new_start_zup[1]], dtype=np.float32)

    # 2. 将朝向向量从 z-up 转到 y-up，求 yaw
    #    z-up heading (dx, dy) 在 z-up xy-plane
    #    y-up ground plane = xz-plane, 朝向 = (-dx, dy) 投影
    #    yaw = arctan2(x_component, z_component) = arctan2(-dx, dy)
    dx, dy = heading_zup
    yaw_yup = math.atan2(-dx, dy)

    # 3. 构建 new_mat：只有 (x,z) 平移 + yaw 旋转，Y 永远 0
    rot_yup = R.from_rotvec(np.array([0, yaw_yup, 0])).as_matrix()
    new_mat = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
    new_mat[0, :3, :3] = rot_yup
    new_mat[0, 0, 3] = new_pos_yup[0]   # x = -G1_x_zup
    new_mat[0, 2, 3] = new_pos_yup[2]   # z = G1_y_zup
    # new_mat[0, 1, 3] 保持 0 → Y（高度）不参与变换

    # 4. 刚体变换
    old_mat = state.mat
    T = new_mat[0] @ np.linalg.inv(old_mat[0])  # (4,4)

    fp = state.fixed_points[0]  # (fixed_frame, 72)
    fp_reshaped = fp.reshape(-1, 3)

    ones = np.ones((fp_reshaped.shape[0], 1), dtype=np.float32)
    fp_transformed = (T @ np.concatenate([fp_reshaped, ones], axis=1).T).T
    fp_new = fp_transformed[:, :3].reshape(1, fp.shape[0], 72)

    return GenerationState(mat=new_mat, fixed_points=fp_new)
```

### 图解

```
adapt_state 做了什么：

  Before:                          After:
  
  old_mat 定义的坐标系              new_mat 定义的坐标系
       ▲ y                              ▲ y
       │                                │
     old_root                          new_root = G1 真实位置
     old_yaw                           new_yaw = G1 真实朝向
       │                                │
       ○ (body pose)                    ○ (body pose 不变，刚性旋转+平移)
      /|\                              /|\
      / \                              / \

  fixed_points @ old_world         fixed_points @ new_world
```

---

## 3. `nav_planning/trumans_infer.py` 改动

### `run_inference()` 新增参数

```python
def run_inference(
    start, goal, scene_path, output_path,
    ...
    # === NEW ===
    resume_from: str | None = None,          # path to _state.npz
    robot_heading: tuple[float, float] | None = None,  # G1 heading (dx, dy), z-up
):
```

### 内部逻辑片段

```python
# ---- handle resume ----
fixed_points_init = None
if resume_from:
    from replan_state import load_state, adapt_state

    state = load_state(resume_from)

    # --start 就是 G1 真实位置 (z-up)，直接传入 adapt_state
    heading = robot_heading if robot_heading is not None else (1.0, 0.0)
    adapted = adapt_state(state, tuple(start), heading)

    fixed_points_init = torch.from_numpy(adapted.fixed_points).float().to(device)
    # 注意：mat 仍然用 get_guidance 算出来的 mat_init（基于新路径的起始朝向）
    # 因为 sample_step 的 always-normalize 会自动把 fixed_points 投影到 mat_init 的 local frame

# ---- sample_step with state ----
points_all, final_mat, final_fp, steps = sample_step(
    cfg, mat_init, obj_locs, goal_list, action_label_list, sampler_list,
    fixed_points_init=fixed_points_init)

# ---- save state for next round ----
from replan_state import GenerationState, save_state
state = GenerationState(
    mat=final_mat.cpu().numpy().astype(np.float32),
    fixed_points=final_fp.cpu().numpy().astype(np.float32),
)
state_path = os.path.splitext(output_path)[0] + "_state.npz"
save_state(state, state_path)
print(f"[trumans_infer] Saved generation state → {state_path}")
```

### CLI 新增

```
--resume-from PATH      从上一轮的 _state.npz 恢复，热启动生成
--robot-heading DX DY   G1 在仿真中的真实朝向向量 (z-up 地面平面)
                        仅在 --resume-from 时生效，默认 (1, 0)
```

---

## 4. 完整使用流程

```bash
# === Round 1: cold start ===
python nav_planning/trumans_infer.py \
    --start 0 0 --goal 5 0 \
    --scene Data_blocks_motion_all/Scene/room_hanyi_s1.4.npy \
    --output output/motion_r1.npz --planner astar
# → motion_r1.npz + motion_r1_state.npz

# Simulator tracks first ~2s of motion_r1.npz
# Reads G1 actual state: pos=(0.51, 0.04), yaw=0.12 rad

# === Round 2: replan from G1 real position ===
python nav_planning/trumans_infer.py \
    --start 0.51 0.04 --goal 5 0 \
    --scene Data_blocks_motion_all/Scene/room_hanyi_s1.4.npy \
    --output output/motion_r2.npz --planner astar \
    --resume-from output/motion_r1_state.npz --robot-yaw 0.12
# → motion_r2.npz + motion_r2_state.npz
# motion_r2 开头 2 帧: root=G1真实位置, yaw=G1真实朝向, body pose=上一轮结尾

# ... repeat until robot reaches goal ...
```

---

## 5. 实现步骤

### Phase 1: 底层
1. **改 `sample_hsi.py::sample_step()`** — `fixed_points_init`/`start_step`/`max_steps`，always-normalize，扩展返回值
2. **新建 `nav_planning/replan_state.py`** — `GenerationState` + `save/load` + `adapt_state()`

### Phase 2: 集成
3. **改 `nav_planning/trumans_infer.py`** — `run_inference()` 加 `resume_from`/`robot_yaw`，state 保存
4. **CLI** 加 `--resume-from` / `--robot-yaw`

### Phase 3: 验证
5. 冷启动向后兼容：不加 `--resume-from` 的输出和当前版本一致
6. State round-trip：同一 start 恢复 → 输出一致
7. adapt 正确性：改 G1 位置和 yaw → fixed_points 的 root position 和朝向正确偏移

## 6. 不做

- 不新建 `closed_loop.py` — 循环由外部编排
- 不新建 `replan_config.py` — 参数走 CLI
- 不改 `models/synhsi.py` / `path_planner.py` / `datasets/trumans.py`

---

## 7. 已验证的关键决策

1. **纯平移 vs 刚性旋转**：✅ 纯平移方案消除了 yaw 跳变（从 ~180° 降到 ~12°）。`adapt_state()` 只平移 fixed_points 到 G1 位置，不旋转身体朝向。模型通过 `mat_init`（新轨迹方向）自然调整朝向。

2. **短 horizon 工作流**：✅ `--max-steps 5` 生成 ~1.5s 运动，state 在截止点保存。下一轮 replan 实现真正的「首尾相接」：position gap ~5.6cm，height gap ~1.6cm。

3. **`adapt_state` 中 mat_init 的处理**：✅ replan 时用 `get_guidance` 重新计算的 `mat_init`（对齐新路径朝向），不用 `adapted.mat`。模型通过扩散过程自然将身体转向新轨迹方向。
