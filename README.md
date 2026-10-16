# 采购合规审查工作台

探索 RAG、只读 MCP 工具和多 Agent 协作在采购审查中的应用：从资料与采购事实中提取依据，汇总审查建议，再由操作者确认或拒绝。

**开发中的源码公开版。** 工作台默认使用明确标识的 deterministic Fake Model，业务数据与回执为本地合成演示。真实模型适配器和评测代码同时保留，当前演示结果不代表真实模型审查质量；人工动作不会写入外部采购系统。

## 项目内容

- 三个只读 Worker 与 LangGraph 汇合，分离政策、合同和采购事实边界。
- PostgreSQL / pgvector 检索、租户与 Case 授权、MCP 只读工具。
- Run 状态、持久化检查点、中断恢复、Trace、审批审计和幂等模拟回执。
- React 工作台：Case 选择、运行进度、建议、依据、Trace 与人工 Gate。
- Docker Compose 本地编排，以及可重复的合成测试数据与版本化评测合同。

## 产品界面

![采购合规审查产品封面](docs/previews/review-cover.png)

以下为产品设计预览，使用演示数据呈现目标体验；当前实现范围见[交付说明](project/docs/delivery.md)。

**审查总览** · 授权案例、审查队列、风险发现与待办，在同一工作台中展开。

![采购审查总览](docs/previews/review-overview.png)

**证据与判断** · 将事实、条款、原文高亮和人工处理连接在同一视图中。

![采购审查证据核验](docs/previews/review-evidence.png)

**执行与审计** · 直观看到多 Agent 并行、结果汇合、授权边界与人工 Gate。

![多 Agent 执行与审计](docs/previews/review-trace.png)

更多细节、高清原图与本地切换预览见[界面画廊](docs/previews/README.md)。

## 本地启动

环境：Docker Compose v2 或更新版本。源码开发另需 Python 3.12、uv、Node.js 22 或 24、npm。

```bash
docker compose -f project/docker-compose.yml -p v104cs-public up --build -d
```

页面默认位于 `http://127.0.0.1:5173`，API 位于 `http://127.0.0.1:8000`。初始化会在本机生成新的数据库角色凭据与演示访问令牌；按[交付与配置说明](project/docs/delivery.md)读取自己的令牌并填入工作台。

本仓没有复制原环境的数据库、令牌或已保存运行。停止环境使用：

```bash
docker compose -f project/docker-compose.yml -p v104cs-public down
```

该命令保留数据卷。公开副本已通过空数据库启动、浏览器登录及核心审查流程检查，具体范围见[开源准备记录](docs/release-preparation.md)。

## 源码开发与验证

```bash
uv sync --project project/backend --locked
npm ci --prefix project/frontend
python3 scripts/check_public_release.py
npm test --prefix project/frontend
npm run build --prefix project/frontend
```

API 鉴权、数据库供给和更多命令见[交付说明](project/docs/delivery.md)。原内部开发档案、评测运行产物、界面验收截图和设计参考归档没有随本版公开。

## 技术与目录

Python / FastAPI / Pydantic / LangGraph / MCP / PostgreSQL / pgvector；React / TypeScript / Vite；pytest / Vitest / Playwright。

```text
project/backend/           后端、版本化资源与测试
project/frontend/          React 工作台与浏览器测试
project/scripts/           数据库引导、交付与质量检查
project/docker-compose.yml 本地编排
project/docs/              使用与配置说明
scripts/                   公开文件检查
docs/                      开源准备记录与展示素材
```

后续持续完善真实模型效果、场景质量、独立复现和交互展示。参与开发见[贡献说明](CONTRIBUTING.md)，安全问题见[安全反馈](SECURITY.md)。

## 许可证

原创代码使用 [MIT](LICENSE) 许可证。依赖与公开资料的分发边界见[第三方声明](THIRD_PARTY_NOTICES.md)。
