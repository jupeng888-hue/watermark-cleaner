@echo off
REM 一键打包为单个 exe（在 Windows 上运行本脚本）
REM 先执行: pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --onefile --windowed --name "WatermarkCleaner" ^
  --add-data "app\core;core" ^
  app\main.py
echo.
echo 打包完成: dist\WatermarkCleaner.exe
pause
