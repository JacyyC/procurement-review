# 本地交付与配置

本项目是开发中的本地审查工作台。默认模型为 `test_only` Fake Fixture，数据为合成采购场景；confirm/reject 记录本地模拟决定，不形成真实采购审批。

## Docker Compose

从仓库根目录启动独立的本地组合：

```bash
docker compose -f project/docker-compose.yml -p v104cs-public up --build -d
```

组合包含 `db`、一次性 `init`、`api` 和 `web`。只向本机回环开放端口，数据库通过容器网络访问。配置支持 `V104CS_WEB_PORT` 和 `V104CS_API_PORT`；端口冲突时可自行更换，再使用对应端口打开页面。

使用 Colima 等容器虚拟机时，源码目录必须在其文件共享范围内，供数据库启动脚本和本地凭据目录进行绑定挂载。只有独立 `docker-compose` 命令的环境，可将上述命令中的 `docker compose` 替换为 `docker-compose`。

数据库超级用户密码、迁移角色与 `review_runtime` 运行角色凭据均在首次初始化时生成，通过受保护的数据卷或本地文件提供。源码中没有可复用的真实数据库密码。

## 演示访问令牌

初始化会在 `project/.local/delivery/credentials/demo-operator.token` 写入当前本地环境的访问令牌。仅在自己的电脑上读取文件，填入工作台的凭据输入框；不要提交、录屏或放入截图。令牌原文不写在本说明中。

普通 `down` 保留数据；不附带删除数据卷的清理命令。公开副本没有带入任何历史容器数据。

## 源码开发

```bash
uv sync --project project/backend --locked
npm ci --prefix project/frontend
npm run dev --prefix project/frontend
```

前端需要配套 API。`project/.env.example` 提供公开配置示例；实际配置保存在被 Git 忽略的 `.env.local` 或进程环境中，按具体启动器显式加载。应用不会因为存在该文件就自动读取它。

真实模型实验使用独立配置与授权。历史脚本中的私人端点已替换为 `model.example.invalid`；此地址不可调用。默认工作台不会因示例文件而切换为真实模型，也不需提供模型密钥。

## 已知限制

- L-1：角色分离由初始化脚本供给，`review_runtime` 仅承担运行角色职责；自备数据库需要自行核验权限。
- L-2：受控故障与合成恢复测试只证明对应测试合同，不代表生产稳定性。
- Q-1：V1-02 等历史真实模型评测原始产物没有随本版分发，本版不复述其质量读数。
- H-1：部分 holdout 数据和评测规则用于冻结实验，变更输入后应重新登记实验，而非沿用旧结果。
- A-1：`.local/runs` 保存本机评测运行材料，默认被 Git 忽略；离线重放需要自己供给对应产物。

原交付与评测脚本仍保留用于研究，其中有些检查依赖未公开的历史产物。2026-10-16 已在隔离环境通过空数据库 Compose 冷启动、浏览器登录、人工确认、幂等回执及 Trace/审计查询；具体范围见[开源准备记录](../../docs/release-preparation.md)。完整浏览器 E2E、所有恢复场景、真实模型效果和生产部署仍未验证。
