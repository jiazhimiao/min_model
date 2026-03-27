@echo off
setlocal

set SCRIPT_DIR=%~dp0
set DATA_PATH=%~1
set CONFIG_PATH=%SCRIPT_DIR%configs\default_config.json
set PYTHONIOENCODING=utf-8

if "%DATA_PATH%"=="" (
    echo Drag your dataset file onto this BAT file to run.
    echo Supported: .pkl  .csv  .xlsx  .xls
    echo.
    pause
    exit /b 1
)

if not exist "%DATA_PATH%" (
    echo Data file not found:
    echo %DATA_PATH%
    echo.
    pause
    exit /b 1
)

if not exist "%CONFIG_PATH%" (
    echo Config file not found:
    echo %CONFIG_PATH%
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo min_model batch training start
echo ============================================================
echo Data file: %DATA_PATH%
echo Config file: %CONFIG_PATH%
echo Output dir: %SCRIPT_DIR%output\batch
echo.

python "%SCRIPT_DIR%scripts\run_all_models.py" "%DATA_PATH%" --config "%CONFIG_PATH%"
if errorlevel 1 (
    echo.
    echo Training failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Finished. Three models and reports have been generated.
echo Output dir: %SCRIPT_DIR%output\batch
echo ============================================================
pause
