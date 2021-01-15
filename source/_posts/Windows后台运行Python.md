---
title: Windows后台运行Python脚本，开机自动运行
date: 2020-10-14 11:15:08
tags: 
- Python
categories: 常用技巧
---

为了每天自动约洗澡也是拼了:sweat_smile: <!--more-->



## 后台运行python脚本

已知目前有一每天定时预约洗澡的Python脚本shower.py，cmd设置后台运行：

```
pythonw shower.py
```

pythonw.exe是无窗口的Python可执行程序，意思是在运行程序的时候，没有窗口，代码在后台执行。

注意如果像我一样电脑同时安装了Python2 和Python3，需要区分用的是哪个phthonw.exe，最简单的是使用绝对路径：

```
D:\python37\pythonw.exe E:\shower.py
```

可打开任务管理器检查pythonw进程是否已经启动。



## 设置开机自动启动

1、新建批处理文件run_shower.bat：

```
@echo off 
if "%1"=="h" goto begin 
start mshta vbscript:createobject("wscript.shell").run("""%~nx0"" h",0)(window.close)&&exit 
:begin 
::
start /b cmd /k "D:\python37\pythonw.exe E:\shower.py"
```

这段代码可以隐藏批处理运行的窗口。

解释：

> 如果双击一个批处理，等价于参数为空，而一些应用程序需要参数，比如在cmd窗口输入shutdowm -s -t 0,其中-s -t 0就为参数。shutdown为%0，-s为%1，-t为%2，以此类推。
> 第一行我们先跳过，看第二行，表示利用mshta创建一个vbs程序，内容为:createobject("wscript.shell").run(……).
> 如果运行的批处理名为a.bat，在C:\下，那%0代表C:\a.bat，%~nx0代表a.bat。h为参数%1，0表示隐藏运行。
> 由于你双击运行，故第一次批处理%1为空，if不成立，转而运行下一句。然后再次打开自己，并传递参数h，此时if成立，跳转至begin开始运行。
> 这两行很经典，可以使批处理无窗口运行。

2、将bat文件放在开机启动项里：Win+R打开运行窗口，输入shell:startup，将bat文件复制进启动文件夹里。

3、重启测试



参考：https://www.cnblogs.com/nmap/articles/8329125.html