@echo off
echo === S-Parameter Agent 打包为 EXE ===

:: 1. 安装依赖
pip install pyinstaller scikit-rf plotly numpy flask

:: 2. 打包
pyinstaller --noconfirm --onefile --console ^
  --name "SParamAgent" ^
  --add-data "scripts/templates;templates" ^
  --add-data "scripts/config.example.json;." ^
  --add-data "scripts/api_index.json;." ^
  --hidden-import skrf ^
  --hidden-import plotly ^
  --hidden-import numpy ^
  --hidden-import flask ^
  scripts/app.py

echo === 完成 ===
echo EXE 在: dist\SParamAgent.exe
echo 运行: dist\SParamAgent.exe
echo 浏览器打开: http://localhost:5050
pause
