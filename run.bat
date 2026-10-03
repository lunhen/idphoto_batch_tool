@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "APP_PY=%~dp0.venv\Scripts\python.exe"
if exist "%APP_PY%" goto check_runtime

rem Select a full Python installation with Tk, not an embedded runtime.
set "BASE_PY="
call :probe py -3.12
if not defined BASE_PY call :probe py -3.11
if not defined BASE_PY call :probe py -3
if not defined BASE_PY call :probe python
if not defined BASE_PY (
  echo [错误] 未找到包含 tkinter 的完整 Python。请安装 Python 3.12 并勾选 Tcl/Tk 和 pip。
  goto failed
)
%BASE_PY% -m venv .venv
if errorlevel 1 goto failed

:check_runtime
"%APP_PY%" -c "import tkinter, venv; r=tkinter.Tk(); r.withdraw(); r.update(); r.destroy()"
if errorlevel 1 (
  echo [错误] 项目运行环境缺少可用的 Tcl/Tk。请修复完整 Python 安装后重建 .venv。
  goto failed
)
"%APP_PY%" -c "import cv2, numpy, PIL" >nul 2>nul
if not errorlevel 1 goto ready
echo 正在安装项目依赖，首次安装需要联网……
"%APP_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto failed

:ready
if /i "%~1"=="--check" exit /b 0
if /i "%~1"=="--prepare" exit /b 0
echo 正在启动证件照生成器……
"%APP_PY%" app.py
if errorlevel 1 goto failed
exit /b 0

:probe
%* -c "import tkinter, venv, ensurepip" >nul 2>nul
if not errorlevel 1 set "BASE_PY=%*"
exit /b 0

:failed
echo.
echo [启动失败] 请保留上方报错信息，以便排查。
if /i not "%~1"=="--check" pause
exit /b 1
