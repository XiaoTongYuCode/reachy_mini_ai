问题点很明确：你用的是 macOS，`reachy-mini-daemon --sim` 会触发 MuJoCo，但 MuJoCo 在 macOS 上要求用 `mjpython` 启动，所以 daemon 后端没起来。后面的 app 报 `Unable to establish connection` 只是结果。

正确启动方式：

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

我刚才验证过 `--mockup-sim --no-media` 能正常启动 daemon，SDK 读到的状态是 `state=RUNNING`、`mockup_sim_enabled=True`。

另外：我这边测试 MuJoCo 时留下了一个 macOS `U` 状态进程 `26120`，`kill -9` 也杀不掉，这是系统调用卡住状态。它没有成功监听 8000，但如果你后续发现奇怪残留，最直接是关掉相关终端，仍不消失就重启机器。