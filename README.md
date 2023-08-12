# yewentao256's Personal Blog

English | [简体中文](README.zh-cn.md)

## Preview

![image](static/resources/website-preview.png)

My personal blog (website example): [https://wentao.site/], welcome to fork and star

## Fork Guide

File directory structure introduction

```bash
yewentao256.github.io
├── archetypes      # Default template for new posts
├── config.toml     # Hugo configuration document
├── content         # Store all articles in Markdown format
├── static          # Store images, css, js, and other resources
└── themes          # Themes
```

1. Set the project name and branch required for gitpage
   It should be `username.github.io`'s main branch or master branch

2. Modify the configuration
   Update the configuration in `config.toml`, and change the personal information to your own.

3. Delete old articles and images
   Delete my articles and images.

4. Set up your domain(optional)
   Set your domain on the GitHub `settings-Pages` page.

5. Start writing your blog
   `hugo new posts/directory/title/index.en.md`
   This command creates a new article with `archetypes/default.md` as the template, located in the `content/posts` directory. After writing, delete `draft: true`.

6. Start the local service for preview
   `hugo server -e production`

## Frequently Asked Questions

- There are two ways to index static files. One is to put them in the post folder, for example:

    ```bash
    # content/posts/pytorch/deep_dive_to_autograd_2

    ❯ ls
    index.en.md     index.zh-cn.md  resources
    ```

    Then you can index in the markdown with a relative path, e.g., `![image](resources/graph.png)`

    Another form is to put them in the **static** folder, for example:

    ```bash
    # static/csapp/resources
    ❯ ls
    ROP-attack.png                                  io-redirection.png
    ```

    Then you can index in any document as `![image](/csapp/resources/ROP-attack.png)`

- If you start the loveit theme on a Mac, you may encounter a fatal error due to the file number limit, fix it with `sudo launchctl limit maxfiles 100000 500000`

- No CSS style after `hugo`? Press F12 to check where the frontend goes to find CSS, and fix `baseUrl` according to related information.
  - For example, locally it may use an absolute path to find it, `/Users/yewentao/Desktop/myblog/docs`
  - After deploying to gitpage, use pageurl to find, like `https://yewentao256.github.io/blog`

## Acknowledgements

- Thanks to **hugo**, an efficient and easy-to-use static page generation tool: [https://github.com/gohugoio/hugo]
- Thanks to **LoveIt**, a neat and elegant hugo theme: [https://github.com/dillonzq/LoveIt]
- Thanks to **Doit**, who continued to maintain this hugo theme after the LoveIt project stopped: [https://github.com/HEIGE-PCloud/DoIt]
