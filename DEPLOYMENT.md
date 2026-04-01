# 部署指南

## 上传到 GitHub

### 1. 创建 GitHub 仓库
在 GitHub 上创建一个新仓库（比如 `AIE1902-Ass1`）

### 2. 初始化并上传

```bash
cd /Users/dyl/Downloads/AIE1902-Ass1-main_副本

# 初始化 git 仓库
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit with Kronos models"

# 连接远程仓库（替换 YOUR_USERNAME 和 REPO_NAME）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 推送
git branch -M main
git push -u origin main
```

## 新电脑运行步骤

### 1. 克隆仓库
```bash
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行
```bash
streamlit run app.py
```

## Kronos 模型说明

模型文件已包含在 `kronos_weights/` 目录中：
- `kronos-small/` - 94MB，推荐日常使用
- `kronos-mini/` - 16MB，轻量级选项
- `tokenizer-base/` - 15MB，tokenizer 文件
- `tokenizer-2k/` - 15MB，tokenizer 文件（给 kronos-mini 使用）

**注意**：代码会自动检测本地模型，无需联网下载！

## 文件大小
总大小约 140MB，GitHub 免费版限制单个文件 100MB，所有文件都在限制内。
