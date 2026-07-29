# OX v0.2 仿真平台

OX v0.2 是一个基于 Python 的车联网（VANET）仿真平台，用于模拟车辆、基站、无人机和云服务之间的通信，以及恶意车辆识别和信誉更新过程。

## 主要功能

- 基于 OpenStreetMap/OSMnx 加载道路网络
- 车辆移动与车联网通信仿真
- 基站和无人机辅助通信
- 信道路径损耗与阴影衰落模拟
- 恶意车辆、反馈与信誉机制仿真
- 动画、帧级统计和汇总结果导出

## 环境要求

- Python 3.10 或更高版本
- 首次加载地图时需要网络连接

## 安装

```bash
python -m pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

使用其他配置文件：

```bash
python main.py --config path/to/config.json
```

仿真参数可在 `config.json` 中调整。默认运行会生成动画和 CSV 统计文件，这些生成文件不会提交到 Git 仓库。

## 核心文件

- `main.py`：程序入口及结果导出
- `sim_core.py`：仿真核心引擎
- `config.json`：默认仿真配置
- `config_loader.py`：配置加载
- `graph_loader.py`：道路网络加载与处理
- `vehicles.py`：车辆模型
- `trust_entities.py`：信任实体与无人机模型
- `trust_experiments.py`：信任机制实验
- `viz.py`：仿真可视化
- `requirements.txt`：Python 依赖

## 输出文件

运行后通常会生成：

- `vanet_sim.gif`
- `frame_stats.csv`
- `summary.csv`
- `trust_events.csv`
- `trust_updates.csv`

## 注意事项

较大规模实验可能需要较长运行时间。建议在修改参数后保存独立配置文件，以便复现实验。
