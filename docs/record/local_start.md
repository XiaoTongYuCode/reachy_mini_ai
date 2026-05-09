当前MacOS下正确启动方式：

```bash
cd /Users/xiaotongyu/Downloads/p/reachy_mini_ai
source .venv/bin/activate

mjpython -m reachy_mini.daemon.app.main --sim --no-media
```

另一个终端：

```bash
cd /Users/xiaotongyu/Downloads/p/reachy_mini_ai
source .venv/bin/activate

reachy-mini-conversation-app --no-camera --gradio
```

如果只是先验证 app 链路，不需要 MuJoCo 物理窗口，用这个更稳：

```bash
reachy-mini-daemon --mockup-sim --no-media
```