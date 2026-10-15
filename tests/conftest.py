"""测试环境适配（DSH Windows 文件沙箱）。

沙箱行为：**pytest 在工作区/临时区里新建的目录，在进程退出后会被收回权限**，
之后连列举和删除都会报 ``PermissionError: [WinError 5]``（已用最小实验确认：
手工创建的目录完全正常，新建目录则被锁死）。因此：

- 测试**不依赖** ``tmp_path`` / ``tmp_path_factory``（它们每次都要新建目录），
  需要文件时写进已存在且可写的目录，用完即删；
- 这里把 pytest 两个会枚举目录的清理钩子换成不枚举的实现，避免它去碰
  上一次运行留下的已加锁目录。

这些只影响测试脚手架的临时目录管理，不影响任何被测行为。
"""

import shutil

import _pytest.pathlib as _pathlib
import _pytest.tmpdir as _tmpdir

# 会话开始时「扫描并保留最近 N 个基线临时目录」需要枚举目录；沙箱会拒绝，
# 且真的删除也没有必要（目录本来就指到临时区）。
_tmpdir.cleanup_numbered_dir = (
    lambda rootdir, keep=3, consider_lock_dead_if_created_before=None: None
)

# 用「整树删除（忽略错误）」替换「先 scandir 再逐个删」的实现。
_tmpdir.rm_rf = lambda path, *args, **kwargs: shutil.rmtree(path, ignore_errors=True)

# 会话结束时的死符号链接清理：本仓库不用符号链接，跳过目录枚举。
_pathlib.cleanup_dead_symlinks = lambda root: None
