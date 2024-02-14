---
{"dg-publish":true,"permalink":"/Notion/Tips/docker/","created":"2024-01-04T18:13:47.327+08:00"}
---

- Image
- Container
    - `docker ps` 显示所有运行中的 container
    - `docker ps -a` 显示所有 container，包括 closed
    - `docker rm $(docker ps -q -a)` 关闭所有 container
        - 有时候删除 image 会遇到 iamge used by closed container 的问题，需要这样解决
- Repository
    - 每个 repo 包含多个 tag
    - 每个 tag 对应一个镜像
    - `<仓库名>:<标签>`
    - 不给出标签默认为 `latest`
- 镜像使用
    - `docker pull [选项] [Docker Registry 地址[:端口号]/]仓库名[:标签]`
        - Docker Registry 地址默认为 Docker Hub (docker.io)
        - 仓库名 一般为两段式名称 `<用户名>/<软件名>`
        - 对于 Docker Hub 如果不给出用户名，则默认为 `library`，也即官方镜像
    - `docker run -it [image name] bash`
        - `-it` 交互式用终端进行操作d
        - `--rm` 运行完毕后将其删除
    - `docker image ls` 列出下载下来的镜像
    - 显示为 `<none>` 的镜像：虚悬镜像，没有用，可以通过 `docker image prune` 删除
    - 删除镜像：`docker image rm [选项] <镜像1> [<镜像2>...]`
        - `<镜像>` 可以是 ID 或 镜像名
- 定制镜像
    - 在一个镜像内部做了一些操作之后，可以通过 `docker diff [容器名]` 看到具体改动
    - 在改动完成后，通过 `docker commit` 命令将其保存称为新镜像
    - `docker commit [选项] <容器ID或容器名> [<仓库名>[:<标签>]]`
    - `-a` 用来指定作者， `-m` 用来记录修改内容，与 git 一样
    - 一般不使用 docker commit 来定制镜像，因为生成的镜像除了想要做的操作之外，可能还有很多其他文件被修改了，而且没有修改记录，无法查看这个镜像具体干了什么
- Dockerfile 定制镜像
    - `FROM [name]` 以一个镜像为基础
        - `FROM scratch` 从空白开始
        - 不以任何系统为基础，直接将可执行文件复制进镜像的做法并不罕见，对于 Linux 下静态编译的程序来说，并不需要有操作系统提供运行时支持，所需的一切库都已经在可执行文件里了，因此直接 `FROM scratch` 会让镜像体积更加小巧。使用 [**Go 语言**](https://golang.google.cn/) 开发的应用很多会使用这种方式来制作镜像，这也是为什么有人认为 Go 是特别适合容器微服务架构的语言的原因之一。
    - `RUN <命令>` or `RUN [“可执行文件”，“参数1”，“参数2”]`
        
        - Dockerfile 中每一个指令都会建立一层，因此一些命令可以叠在一层里
        
        ```Docker
        FROM debian:stretch
        
        RUN apt-get update
        RUN apt-get install -y gcc libc6-dev make wget
        RUN wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz"
        RUN mkdir -p /usr/src/redis
        RUN tar -xzf redis.tar.gz -C /usr/src/redis --strip-components=1
        RUN make -C /usr/src/redis
        RUN make -C /usr/src/redis install
        ```
        
        应当写为
        
        ```Docker
        FROM debian:stretch
        
        RUN set -x; buildDeps='gcc libc6-dev make wget' \
            && apt-get update \
            && apt-get install -y $buildDeps \
            && wget -O redis.tar.gz "http://download.redis.io/releases/redis-5.0.3.tar.gz" \
            && mkdir -p /usr/src/redis \
            && tar -xzf redis.tar.gz -C /usr/src/redis --strip-components=1 \
            && make -C /usr/src/redis \
            && make -C /usr/src/redis install \
            && rm -rf /var/lib/apt/lists/* \
            && rm redis.tar.gz \
            && rm -r /usr/src/redis \
            && apt-get purge -y --auto-remove $buildDeps
        ```
        
        运行完之后清理掉无关文件
        
    - 构建镜像
        - `docker build [选项] <上下文路径/URL>`
        - 一般是在 Dockerfile 所在目录执行 `docker build -t [name:version] .`
        - 最后的 `.` 表示上下文目录，在 Dockerfile 中有许多对文件的操作，比如复制等，就是根据与上下文目录的相对路径访问文件，只能访问这个目录下的文件
        - 在 Dockerfile 创建镜像的时候，会把整个上下文目录打包并且供给创建过程使用，因此需要把要用到的东西复制进来，而不是直接把 Dockerfile 放到一个很高级的目录中去，以防止需要打包巨量文件
        - 可以通过 .dockerignore 来剔除不需要传递给 Docker 引擎的文件
        - 可以通过 `-f 文件名` 来指定 `Dockerfile`
    - [Dockerfile 详解](https://vuepress.mirror.docker-practice.com/image/dockerfile/)
- 其他
    - `docker login` 登录进一个 docker 账号
    - 进到一个 docker 里面之后先把 .ssh 复制到 ~ 目录下
- 使用 docker run -v 进行本地目录挂载
    - `docker run -v /home/zongtai:/mnt ...`
    - 把 `/home/zongtai` 目录挂载到 `/mnt` ，两个目录同步更改