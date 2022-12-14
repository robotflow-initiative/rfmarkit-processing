# 启动测量

## 概述

- `measurement_server.py` 启动一个测量服务器
- `test_api.py` 对服务器API的测试

## `measurement_server.py`

开启侦听

```console
python measurement_server.py -l --port <串口端口>
```

脚本会创建两个管道`/tmp/measurement_pipe_in`和`/tmp/measurement_pipe_out`，管道的名称可以用`--pipe_in`和`--pipe_out`修改。

> - `/tmp/measurement_pipe_in` 用来读取控制命令
> - `/tmp/measurement_pipe_out` 用来返回执行结果

如果不使用`-l`参数，将会单独开始一次记录。记录的名称可以通过`--file`进行指定。

```console
python measurement_server.py --file <记录名称>
```

`measurement_server.py` 中的`record_csv`函数是记录的核心代码。

## 与测量服务器进行交互

与测量服务器进行交互的方法是通过UNIX管道。命令的格式是json字符串，例如

```console
echo "{\"type\":\"start\"} > /tmp/measurement_pipe_in
```

或者

```console
echo "{\"type\":\"start\", \"measurement_filename\":\"test.csv\"} > /tmp/measurement_pipe_in
```

就可以开始记录。

而

```console
echo "{\"type\":\"stop\"} > /tmp/measurement_pipe_in
```

可以停止记录。

> 重复的start命令会被拦截，因为目前还上尚不支持从多个串口连接的IMU记录数据


## `test_api.py`

`test_api.py` 演示了开始记录，停止记录和退出的三个API
