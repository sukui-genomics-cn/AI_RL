# Proxy Introduce



## Kinds of Proxy

**HTTP**

HTTP 代理在客户端浏览器和web服务器之间从当中间人的角色, 它接收来自客户端的HTTP请求, 并将其转发给web服务器, 然后将服务器的响应传回给客户端. HTTP代理具备解析和处理HTTP流量的能力, 因此可以执行诸如缓存网页以提高后续访问速度, 过滤内容和管理数据流的任务, 还未用户提供了一定程度的匿名性, 并可以用于绕过地理限制.

**Socks**

SOCKS是一种网络协议, 通过代理服务器路由网络流量, 使客户端能够连接到服务器. 她在OSI模型的第5层)会话层)操作, 处于表示层和传输层之间. SOCKS代理用于一般目的, 在涉及大量流量任务中特别好用. 

- 支持TCP和UDP协议, 使用SOCKS能够处理更多种类的网络流量, 而不仅限于网页请求, 例如DNS查询和P2 P共享
- 多种身份验证方法, 可以通过用户名和密码或更复杂的方法来保护代理服务器
- 执行远程DNS查询能力, 允许通过代理服务器DNS请求, 从而防止潜在的DNS泄漏, 增强隐私保护.

**DNS Server**

DNS 是域名系统简称, 因特网上作为域名和IP地址相互映射的一个分布式数据库.

以访问baidu.com为例, DNS会进行一下操作: 主机名.次级域名.顶级域名.跟域名 -> www.baidu.com.root

- 查找电脑上缓存的DNS缓存列表, 如果有, 那么直接放回对应的IP地址
- 查找电脑上HOST文件的映射关系, 如果有, 则返回对于IP地址
- 查找互联网线路供应商的本地DNS服务器(电信, 移动,..), 本地DNS服务器先查找自己的缓存记录, 如果有记录, 那么返回对应IP地址, 否则本地服务器向根域名服务器发生请求.
- 根域名服务器收到请求后, 查看是.com顶级域名, 于是返回.com顶级域名的ip地址给到本地DNS服务器
- 本地DNS服务器收到回复后, 向.com顶级域名服务器发起请求
- .com顶级域名服务器收到请求后, 查看是.baidu.com次级域名, 于是返回.baidu.com次级域名
- 本地DNS服务器收到baidu.com的IP后, 向电脑回复域名对应ip, 并把记录写入本地DNS服务器的缓存例

![](https://pica.zhimg.com/v2-baaa52c463ce10522735c49a3239399a_1440w.jpg)

**DNS代理**

DNS代理用于DNS Client和DNS server之间转发DNS请求和应答报文. 局域网内的DNS client把DNS proxy当作DNS server, 将请求报文发送给DNS Proxy. DNS Proxy将该请求报文转发至DNS Server, 并将DNS server的应答报文返回给DNSClient, 从而实现域名解析.

使用DNS Proxy后, 当DNS Server地址发生变化时, 只需要改变DNS Proxy上的配置,无需改变局域网内每个DNS Client的配置, 从而简化了网络管理.

## Python

HTTP 全局代理设置

```bash
export http_proxy="http://127.0.0.1:1231"
export https_proxy="http://127.0.0.1:1231"
```

**SOCKS全局代理**

通过设置环境变量的方式通常只能使用HTTP代理, 要使用全局SOCKS代理, 可以使用tsocks

安装tsokcs后, 编辑`/etc/tsokcs.conf` , 使用端口为8080的本地socks5代理为例

```bash
server=127.0.0.1
server_port=8080
server_type=5
```

配置完成后, 在原来的脚本执行命令钱添加tsocks即可使用

```bash
tsocks python script.py
```

**针对部份请求设置代理**

前面的几种方式会为所有HTTP请求设置代理, 部份使用代理可以使用requests的proxies参数

```python
import requests
proxies = {
    "http":"socks:5//127.0.0.1:8080",
    "http":"socks:5//127.0.0.1:8080",
}
```



## Git

```bash
# git config
git config --global user.name "kuisu"
git config --global user.email "kuisu_dgut@163.com"
git config -l # show the config

# git local config -> 仅对当前仓库有效
git config --local user.name "kuisu"

# git 配置代理
git clone https://huggingface.co/Qwen/QwQ-32B --config "http.proxy=192.168.72.117:1082"

# if you want to clone without large file
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/QwQ-32B

# 拉去远程分支合并
# git pull <远程主机名> <远程分支名>:<本地分支名>
git pull origin next:maxter

# 类似于
git fetch origin
git merge origin/next
```



## Browser



## Local area Network