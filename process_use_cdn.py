import os
import time

def process_resources_use_cdn(path: str = "content/posts") -> None:
    """本函数将处理所有markdown文件, 将图片文件引用转化为cdn引用, 实现本地部署到线上的转换
    
    例如：
    `![image](resources/android-chrome-192x192.png)
    将转换为：
    `![image](https://cdn.jsdelivr.net/gh/yewentao256/blog/content/posts/first-post/resources/android-chrome-192x192.png)`

    Args:
        path (str): post 根目录
    """
    now = time.time()
    cdn_prefix = '![image](https://cdn.jsdelivr.net/gh/yewentao256/blog/'
    for root, _, files in os.walk(path):
        # 替换字符
        for file_name in files:
            if not file_name.endswith('md'):
                continue
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f'processing {file_path} ...')
                content = f.readlines()
                for i, line in enumerate(content):
                    if '![image](resources/' in line:
                        content[i] = line.replace('![image](resources/', cdn_prefix + root.replace('\\', '/') + '/resources/')
                        print(f'    changing `{line}` to `{content[i]}`'.replace('\n', ''))
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(content)
    print(f'Done! time usage: {time.time() - now}')

if __name__ == '__main__':
    process_resources_use_cdn()