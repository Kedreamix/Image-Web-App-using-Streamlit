# 神奇的streamlit (原来深度学习还可以这样玩)

[toc]
&ensp;&ensp;&ensp;&ensp;讲道理，我已经好久没有看到这么这么好玩的东西了哈哈，有时候好玩的东西或者是学习成果要快速展示出来，不是一件容易的事情。辛辛苦苦把核心算法实现了，还要做一个前端页面。不过今天试用了一下Python领域发展迅速的一个开源项目Streamlit，它能帮你不用懂得复杂的HTML，CSS等前端技术就能快速做出来一个炫酷的Web APP。（超级帅）

## Streamlit 简介

&ensp;&ensp;&ensp;&ensp;Streamlit是一个基于Python的可视化工具，和其他可视化工具不同的是，它生成的是一个可交互的站点（页面）。但同时它又不是我们常接触的类似Django、Flask这样的WEB框架。

它有一个优点：

- 无需编写任何HTML、CSS或JS代码就可以生成界面不错的页面

&ensp;&ensp;&ensp;&ensp;streamlit是一个机器学习工程师专用的应用程序框架



## Streamlit带来的改变

原先的数据展示页面开发流程：

1. 在Jupyter中开发演示
2. 将Python代码复制到文件
3. 编写Flask应用，包括考虑HTTP请求、HTML代码、JS和回调等

![img](https://img-blog.csdnimg.cn/img_convert/87e056ca8da4587d2604ee8f3d607cb8.png)

而当展示页面非常重要时，通常的流程是这样的：

1. 收集用户需求
2. 定义展示框架与原型
3. 使用HTML、CSS、Python、React、Javascript等进行编码
4. 一个月以后才能看到最终的页面

![img](https://img-blog.csdnimg.cn/img_convert/8689709712ffac53d1fe94cdb0f05652.png)

Streamlit的流程：

- 稍微改下Python代码即可生成展示界面

![\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-8xZDZBDk-1638941769333)(C:\Users\86137\Desktop\flow-3.png)\]](https://img-blog.csdnimg.cn/c270f3c1ba7340e98204362b32a9ffc9.png)


&ensp;&ensp;&ensp;&ensp;而能够快速生成应用，主要原因是Streamlit兼容以下Python库或框架：

- 数据处理：Numpy、Pandas
- 机器学习框架或库：Scilit-Learn、TensorFlow、Keras、PyTorch
- 数据可视化工具：matplotlib、seaborn、poltly、boken、Altair、GL、Vega-Lite
- 文本处理：Markdown、LaTeX

## Streamlit的简单应用

&ensp;&ensp;&ensp;&ensp;你是否还是为不懂深度学习而烦恼，你是否不知道不同的深度学习参数有什么区别，你是否还是思考什么优化器比较好，你是否不太知道激活函数和损失函数的选择which better，你是否还在思考要去哪里找代码并且去在编译器上跑出一个好丑的结果，那就快看Streamlit，他可能让本身你不懂机器学习的同学也能测试出很好的结果哦，体验深度学习的魅力。

&ensp;&ensp;&ensp;&ensp;

&ensp;&ensp;&ensp;&ensp;我这里呢，也就是一个简单的用三层神经网络模型，对较为经典的MINST数据进行分类，我们可以在可视化模型，调整各个参数，然后就能在一个好看的页面上显示我们的结果，更加简便快捷的部署在我们的云端上。

首先，我们打开我们的命令行，在当前目录下输入，demo.py就是我们写好的python文件

```python
streamlit run demo.py
```

然后我们就可以看到这个界面，接着我们可以从我们的本地端口local进行打开

![在这里插入图片描述](https://img-blog.csdnimg.cn/77c372b6365348baaae2a1c257cb1c12.png)



接着打开我们的localhost:8501，就会显示编写好的页面，我们可以看到这是一个3层的神经网络，我们利用手写数字数据集MINST来训练，从下图中我们可以看出，我们可以调整

- **迭代次数 epoch**
- **每次处理的批次大小 batch size**
- **输出的类别 output classes**
- **隐藏层的节点数 hidden nodes**
- **优化器 optimizer**
- **激活函数 Activation**
- **损失函数 Loss function**

![在这里插入图片描述](https://img-blog.csdnimg.cn/9d6832ab83084eacaea2b14d274a248b.png)

&ensp;&ensp;&ensp;&ensp;除此之外，我们还可以可视化我们的model，我们只需要在左侧勾选我们的复选框即可得到。

![在这里插入图片描述](https://img-blog.csdnimg.cn/ed7019f80c374095b268f54d51df7ded.png)



&ensp;&ensp;&ensp;&ensp;然后我们设置好我们的参数，点击我们的Process就可以运行我们的结果了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/9f03523292764ecebde0f41435910db6.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/81ecad9d74b94a9a88affa6660f7cf29.png)



是不是超级好玩，这样即使你不懂，也可以进行调参哦

## Streamlit部署计算机视觉（图像分类和目标检测）

&ensp;&ensp;&ensp;&ensp;除此之外，还可以用已有的模型，部署到我们的网页上，这样我们一下子就能用了哦

&ensp;&ensp;&ensp;&ensp;这里给出结果看看

### 图像分类部署

&ensp;&ensp;&ensp;&ensp;我们可以直接上传我们的图片，然后就会在后端运行，将得到的结果返回

![在这里插入图片描述](https://img-blog.csdnimg.cn/8acd5d1c9c64411a9d5193221b335430.png)

### 目标检测

&ensp;&ensp;&ensp;&ensp;目标检测也是一样的

![在这里插入图片描述](https://img-blog.csdnimg.cn/3231ce9d3a4544e4bfbedefacbbab0b7.png)

&ensp;&ensp;&ensp;&ensp;看！！！是不是很好玩，我突然觉得我做的很多东西都有用武之地咯



**每日一句**

**You never know how strong you are until being strong is your only choice.**

**你永远不知道自己有多强大，直到变强是你唯一的选择。**
