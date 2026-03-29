@echo off
set PYTHON=C:\Users\dingj\AppData\Local\Programs\Python\Python312\python.exe
set PROJECT=C:\Users\dingj\Project\NBA-Simulverse
set LOGDIR=%PROJECT%\logs

cd /d %PROJECT%
%PYTHON% scripts\16_daily_picks.py >> %LOGDIR%\picks_cron.log 2>&1
