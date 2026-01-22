# 🚀 快速开始 - Motion Tracker

## ⚡ 3分钟运行第一个Demo

### 1. 安装依赖

```bash
cd /Volumes/MindDockSSD/projects/opensource/motion-tracker

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装包
pip install mediapipe opencv-python numpy
```

### 2. 下载模型（首次必须）

选择一个模型下载（推荐 Full）:

```bash
# 创建模型目录
mkdir -p models

# Full 模型（推荐，平衡性能和精度）
curl -L -o models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
```

### 3. 运行！

```bash
python demos/webcam_demo.py
```

按 `q` 退出，按 `s` 截图。

## 🎮 四个演示程序

### 1️⃣ 实时姿态检测

```bash
python demos/webcam_demo.py --show-fps
```

显示：
- 33个关键点
- 实时角度计算
- FPS 计数器

### 2️⃣ 坐姿矫正

```bash
python demos/posture_correction_demo.py
```

功能：
- 按 `c` 校准正确坐姿
- 实时提示姿态问题
- 绿色=正确，红色=需要矫正

### 3️⃣ AI健身教练

```bash
python demos/fitness_trainer_demo.py
```

支持运动：
- `1` = 深蹲
- `2` = 俯卧撑
- `3` = 二头肌弯举
- `4` = 肩推

自动计数重复次数，实时反馈动作质量。

### 4️⃣ 舞蹈教练

```bash
python demos/dance_coach_demo.py
```

工作流程：
1. 按 `r` 开始录制参考动作（跳一段舞）
2. 按 `r` 停止录制
3. 按 `p` 开始练习模式
4. 跟着跳，系统实时对比并打分
5. 按 `p` 停止，查看总分

特色：
- DTW 时序对齐（速度快慢都能匹配）
- 8个关节实时对比
- 100分制评分
- 可保存参考动作（按 `s`）

## 🔧 常见问题

### ❌ 摄像头打不开

```bash
# 尝试其他摄像头
python demos/webcam_demo.py --camera 1
```

### ❌ 模型下载失败

手动下载（见上方步骤2）或查看 `docs/INSTALLATION.md`

### ❌ FPS 太低

```bash
# 使用轻量模型（下载 lite 模型）
# 降低分辨率
python demos/webcam_demo.py --width 640 --height 480
```

## 📚 更多文档

| 文档 | 说明 |
|------|------|
| [README.md](README.md) | 完整项目文档 |
| [QUICKSTART.md](QUICKSTART.md) | 5分钟教程 |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | 详细安装指南 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构设计 |
| [STATUS.md](STATUS.md) | 项目状态 |

## 💡 下一步

1. ✅ 运行三个 demo 熟悉功能
2. 📖 阅读 API 文档学习如何集成
3. 🛠️  修改代码实现自己的应用
4. 🌟  Star 项目并分享！

## 🆘 需要帮助？

- Issues: https://github.com/MindDock/motion-tracker/issues
- Email: your.email@example.com

Happy Tracking! 🏃‍♂️💪🤸‍♀️
