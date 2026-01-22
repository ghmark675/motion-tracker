# 骨架渲染改进说明

## 🎯 改进内容

### v0.1.3 - 2026-01-22

#### 1. 添加颈部连接线 ✅

**之前**:
- 头部（鼻子、耳朵）与身体分离
- 没有颈部区域的视觉连接
- 看起来头是"漂浮"的

**现在**:
```
新增连接：
  - nose → left_shoulder
  - nose → right_shoulder
  - left_ear → left_shoulder
  - right_ear → right_shoulder
```

**效果**:
- 完整的人体骨架
- 颈部姿态可视化
- 头部与身体的连接关系清晰

#### 2. 增强角度显示 ✅

**之前**:
- 小字体角度文本
- 所有关节都显示（太杂乱）
- 角度文本可能被忽略

**现在**:
```python
改进：
  - 只显示主要关节（8个）
  - 字体更大（0.6 vs 0.5）
  - 粗体文本（thickness=2）
  - 彩色高亮圆圈标注关节
  - 文本带背景（易读）
```

**显示的关节**:
- Left/Right Elbow（肘关节）
- Left/Right Shoulder（肩关节）
- Left/Right Knee（膝关节）
- Left/Right Hip（髋关节）

#### 3. 视觉编码改进 ✅

**颜色系统**:
```
角度范围（肘部/膝部）:
  160-180° → 绿色 (良好伸展)
  140-160° → 橙色 (中等)
  < 140°   → 红色 (弯曲)
```

**视觉元素**:
- 黄色连接线（2像素粗）
- 绿色关键点（4像素半径）
- 彩色角度圆圈（8像素半径）
- 白色/彩色角度文本

#### 4. 字符编码修复 ✅

**问题**: 终端显示 `?` 而不是 `°`

**修复**:
- 全局替换 `°` → `deg`
- 确保 ASCII 兼容性
- 所有终端都能正常显示

---

## 📊 完整骨架连接列表

### 总计：31 条连接线

#### 头部和颈部 (6条) - NEW!
```
1. nose → left_shoulder
2. nose → right_shoulder
3. left_ear → left_shoulder
4. right_ear → right_shoulder
5. nose → left_eye
6. nose → right_eye
```

#### 躯干 (4条)
```
7. left_shoulder ↔ right_shoulder
8. left_hip ↔ right_hip
9. left_shoulder → left_hip
10. right_shoulder → right_hip
```

#### 左臂 (5条)
```
11. left_shoulder → left_elbow
12. left_elbow → left_wrist
13. left_wrist → left_thumb
14. left_wrist → left_index
15. left_wrist → left_pinky
```

#### 右臂 (5条)
```
16. right_shoulder → right_elbow
17. right_elbow → right_wrist
18. right_wrist → right_thumb
19. right_wrist → right_index
20. right_wrist → right_pinky
```

#### 左腿 (4条)
```
21. left_hip → left_knee
22. left_knee → left_ankle
23. left_ankle → left_heel
24. left_ankle → left_foot_index
```

#### 右腿 (4条)
```
25. right_hip → right_knee
26. right_knee → right_ankle
27. right_ankle → right_heel
28. right_ankle → right_foot_index
```

#### 面部细节 (3条)
```
29. left_eye → left_ear
30. right_eye → right_ear
31. mouth_left ↔ mouth_right
```

---

## 🎨 视觉效果对比

### 之前
```
问题：
❌ 头部与身体分离
❌ 角度文本太小
❌ 显示过多信息（杂乱）
❌ 特殊字符显示问题
```

### 现在
```
改进：
✅ 完整的人体骨架（包括颈部）
✅ 清晰的角度标注（8个主要关节）
✅ 彩色编码（绿/橙/红）
✅ 高亮圆圈标注
✅ ASCII 友好（deg 而非 °）
```

---

## 🚀 使用示例

### 运行 webcam demo

```bash
cd motion-tracker
source venv/bin/activate
python demos/webcam_demo.py --show-fps
```

### 你应该看到：

1. **黄色骨架线**:
   - 完整的身体连接
   - **颈部连接清晰可见**
   - 手指和脚趾的细节

2. **绿色关键点**:
   - 33个检测点
   - 小圆圈标记

3. **角度标注** (8个主要关节):
   ```
   左肘: 168deg [绿色圆圈]
   右肘: 165deg [绿色圆圈]
   左肩: 45deg  [橙色圆圈]
   右肩: 47deg  [橙色圆圈]
   左膝: 175deg [绿色圆圈]
   右膝: 173deg [绿色圆圈]
   左髋: 170deg [绿色圆圈]
   右髋: 168deg [绿色圆圈]
   ```

4. **两个信息面板**:
   - 左上：姿态指标（头部倾斜、身体倾斜等）
   - 右上：置信度、FPS、关节角度

---

## 🔧 技术细节

### 渲染顺序

```python
1. 绘制连接线 (_draw_connections)
   - 遍历 31 条连接
   - 检查可见性
   - 画黄色线条

2. 绘制关键点 (_draw_keypoints)
   - 33个绿色圆圈
   - 可选标签

3. 绘制角度 (_draw_angles)
   - 只显示8个主要关节
   - 彩色高亮圆圈
   - 角度文本 + 背景
```

### 关键代码

```python
# 颈部连接（新增）
CONNECTIONS = [
    ('nose', 'left_shoulder'),   # 鼻子到左肩
    ('nose', 'right_shoulder'),  # 鼻子到右肩
    ('left_ear', 'left_shoulder'),   # 左耳到左肩
    ('right_ear', 'right_shoulder'), # 右耳到右肩
    # ... 其他连接
]

# 角度显示（增强）
major_joints = [
    'left_elbow', 'right_elbow',
    'left_shoulder', 'right_shoulder',
    'left_knee', 'right_knee',
    'left_hip', 'right_hip',
]

# 绘制高亮圆圈
cv2.circle(frame, (x, y), 8, color, 2)

# 绘制角度文本（更大更粗）
self._draw_text_with_background(
    frame, text, (x+15, y+5), color,
    font_scale=0.6,  # 更大
    thickness=2,     # 更粗
)
```

---

## 📈 性能影响

### 渲染性能

```
改进前: ~35 FPS
改进后: ~35 FPS

影响: 无明显影响
原因:
  - 减少了角度显示数量（杂乱 -> 8个）
  - 连接线数量增加很少（27 -> 31）
  - 绘图操作高度优化
```

---

## ✅ 验证检查清单

运行 webcam demo 时，确认你能看到：

- [ ] 黄色骨架线连接全身
- [ ] **颈部线条**（鼻子/耳朵到肩膀）
- [ ] 绿色圆圈标记关键点
- [ ] 8个主要关节的角度文本
- [ ] 彩色圆圈高亮显示关节
- [ ] 角度文本显示 "deg" 而不是问号
- [ ] 左上面板显示姿态指标
- [ ] 右上面板显示关节角度

---

## 🎓 常见问题

### Q: 为什么只显示8个关节的角度？

A: 显示所有关节会导致画面过于杂乱。8个主要关节（肘、肩、膝、髋）是最重要的运动指标。

### Q: 如何显示所有角度？

A: 修改 `_draw_angles` 方法中的 `major_joints` 列表，添加更多关节名称。

### Q: 颈部线条太多了？

A: 可以在 CONNECTIONS 列表中注释掉不需要的连接，但建议保留至少 `nose → shoulders` 来显示颈部。

### Q: 如何改变颜色？

A: 修改 `SkeletonRenderer.COLORS` 字典：
```python
COLORS = {
    'connection': (0, 255, 255),  # 黄色 -> 改成其他颜色
    'keypoint': (0, 255, 0),      # 绿色
    # ...
}
```

---

## 📝 未来改进

可能的进一步优化：

- [ ] 根据置信度调整线条透明度
- [ ] 添加骨架运动轨迹
- [ ] 3D 视角切换
- [ ] 关节角度历史图表
- [ ] 可配置的显示选项（GUI）
- [ ] 导出带标注的视频

---

**最后更新**: 2026-01-22
**版本**: v0.1.3
**状态**: ✅ 完成
