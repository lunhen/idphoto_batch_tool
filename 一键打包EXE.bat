@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
call run.bat --prepare
if errorlevel 1 exit /b 1
set "APP_PY=%~dp0.venv\Scripts\python.exe"
"%APP_PY%" -m pip install pyinstaller
if errorlevel 1 goto failed
set "MODEL_ARG="
if exist "models\face_detection_yunet_2023mar.onnx" set MODEL_ARG=--add-data "models\face_detection_yunet_2023mar.onnx;models"
"%APP_PY%" -m PyInstaller --noconfirm --clean --windowed --collect-data cv2 --name "批量二寸证件照生成器" %MODEL_ARG% app.py
if errorlevel 1 goto failed
echo 打包完成，请保留 dist 中生成的整个程序文件夹。
pause
exit /b 0
:failed
echo [打包失败] 请查看上方报错信息。
pause
exit /b 1
