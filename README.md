# yewentao256的个人博客

## 本项目使用说明

基于hugo开发的个人博客项目

- 文件目录结构

```bash
myblog
├── archetypes      # 新文章默认模板
├── config.toml     # Hugo配置文档
├── content         # 存放所有Markdown格式的文章
├── data            # 资源目录
├── layouts         # 存放自定义的view
├── static          # 存放图像、CNAME、css、js等资源，发布后为网页根目录
└── themes          # 主题
```

- 开始书写博客

```bash
hugo new posts/directory/title/index.en.md
```

该命令以`archetypes/default.md`为模板创建新文章，位于`content/posts`目录下，书写好后删除`draft: true`即可

- 本地启动服务

```bash
hugo server -e production
```

- 将本地资源换成cdn资源

```python
python process_use_cdn.py
```

- 构建网页

```bash
hugo
```

## 常见问题

- 如果是在mac上启动loveit主题可能会遇到文件数量限制导致fatal error的问题，`sudo launchctl limit maxfiles 100000 500000`修复

- `hugo`后没有css样式？f12查看前端去哪里找css了，根据相关信息修复`baseUrl`
  - 例如：本地可能用绝对路径来找，`/Users/yewentao/Desktop/myblog/docs`
  - 部署到gitpage后用pageurl来找，如`https://yewentao256.github.io/blog`

## Referrence

- [hugo](https://gohugo.io/getting-started/quick-start/)
- [loveit](https://github.com/dillonzq/LoveIt)
