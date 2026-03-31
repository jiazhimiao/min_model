# Git 使用说明

当前目录 `D:\Trae_pro\min_model` 已经是一个从 GitHub 克隆到本地的仓库。

对应远端仓库：

```text
git@github.com:jiazhimiao/min_model.git
```

## 你现在这个动作，学名叫什么

第一次把 GitHub 上的仓库下载到本地，学名叫：

```bash
git clone
```

中文一般叫：

- 克隆仓库
- 把远程仓库克隆到本地

你现在这个项目已经克隆完成了，所以当前目录本身就是本地仓库，不需要再重复 clone 一次。

## 常用 Git 流程

### 1. 第一次获取项目

```bash
git clone git@github.com:jiazhimiao/min_model.git
```

### 2. 进入项目目录

```bash
cd min_model
```

### 3. 查看当前状态

```bash
git status
```

### 4. 拉取远端最新代码

```bash
git pull origin main
```

### 5. 提交本地修改

```bash
git add .
git commit -m "你的提交说明"
```

### 6. 推送到 GitHub

```bash
git push origin main
```

## 你现在这个仓库的状态

- 当前目录已经是 Git 仓库
- 当前分支是 `main`
- 已关联 GitHub 远端 `origin`

## 常见概念

- `clone`：第一次把远端仓库下载到本地
- `pull`：把远端最新更新拉到本地
- `add`：把修改加入暂存区
- `commit`：把本地修改保存成一个版本
- `push`：把本地提交上传到远端

## 一个简单记法

如果你改完代码，最常用的就是这三步：

```bash
git add .
git commit -m "更新说明"
git push origin main
```

如果你只是想先同步远端最新代码：

```bash
git pull origin main
```
