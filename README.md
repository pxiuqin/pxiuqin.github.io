## 简单介绍
Thank you for reading my blog, I will share some technical articles and life ideas at [pxiuqin.github.io](https://pxiuqin.github.io)

## 本地构建
* 在 GitHub 发布自己的站点，参考👉[GitHub Pages介绍](https://pages.github.com/)
* 安装 Jekyll，参考👉[使用 Jekyll 设置 GitHub Pages 站点](https://docs.github.com/cn/pages/setting-up-a-github-pages-site-with-jekyll)
    ```bash
    # 安装 Ruby和开发工具Devkit 

    # 安装 Jekyll
    gem install jekyll bundler

    # 测试
    jekyll -v
    ``` 
* 构建本地测试环境，参考[使用 Jekyll 构建本地测试环境](https://docs.github.com/cn/pages/setting-up-a-github-pages-site-with-jekyll/testing-your-github-pages-site-locally-with-jekyll)
    ```bash
    # 使用Gemfile安装环境依赖
    bundle install
    
    # 启动本地服务
    bundle exec jekyll serve

    # 浏览器查看4000端口服务
    ```

**注意：** 
* 如果您使用的是 Ruby 3.0 和 Jekyll 4.2.x 或更早版本，则需要在运行 bundle install 之前，将 webrick gem 添加到项目的 Gemfile 中
* 如果项目中缺少 Gemfile，使用 `bundle init` 命令生成 

## 使用说明
* 在 `_posts` 目录下添加 Markdown 文件，参考👉[Jekyll Posts 使用](https://jekyllrb.com/docs/posts/)
* 使用`构建本地测试环境`中的内容进行本地测试
* 图片内容一般添加到 `images` 目录中
* 样式修改的话可以处理 `styles` 目录下的 css 文件


