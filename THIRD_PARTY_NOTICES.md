# 第三方声明

本仓原创代码按 MIT 分发。Python 与 npm 依赖保留各自上游许可证，具体版本以 `project/backend/uv.lock` 和 `project/frontend/package-lock.json` 为准。本仓不分发已安装依赖。

主要依赖包括 React、Lucide、Vite、FastAPI、LangGraph、MCP、Pydantic、SQLAlchemy、psycopg、pgvector、pypdf、pytest、Vitest 和 Playwright。数据库与容器镜像同样按各自许可使用。

`project/backend/src/procurement_review/resources/public_sources/manifest.v1.json` 只保存公开资料的来源、校验值、许可说明及获取方法；记录明确声明不随仓库再分发原文。源站资料依其许可使用，不因本仓 MIT 而改变许可。

采购事实、Gold Case、模型 Fixture 与安全反例为合成测试资源。它们不构成真实订单、合同、授权或法律意见。测试中形似个人信息或密钥的反例用于验证脱敏与拒绝边界。

原设计参考图片、前端归档、内部验收截图和模型运行记录未随本版分发。后续添加新素材时应登记来源与分发许可。

`docs/previews/` 的原创页面布局、图表和文字随本仓 MIT 许可分发。基础线性 SVG 图标包含 Feather/Lucide 图形的改写，保留其 [MIT / ISC 许可声明](docs/previews/ICON-LICENSES.txt)。页面使用系统字体，未附带字体文件或外部图库图片；PNG 为这些本地页面的渲染结果，基础图标仍适用其上游许可。
