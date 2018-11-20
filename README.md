### This is the code of book 'Hands-on Machine Learning with Scikit-Learn and TensorFlow' by Aurélien Géron, I annotate it and upload here. thanks Aurélien Géron wrote this book to teach us ML, ckick  [here](https://github.com/ageron/handson-ml) to go to his repository.

### 这个repository是Aurélien Géron所著《Hands-on Machine Learning with Scikit-Learn and TensorFlow》书中的代码，我将代码进行了详细的解释(更新中)。原书作者的repository点 [这里](https://github.com/ageron/handson-ml)
更新状态：
*第一章完结*
*第二章ing*
*第三章ing*

Machine Learning Notebooks
==========================
机器学习notebook
==========================

This project aims at teaching you the fundamentals of Machine Learning in
python. 

这个项目旨在传授你python中机器学习的基础知识。

It contains the example code and solutions to the exercises in my O'Reilly book 

它包含了我书中的代码和练习的解决方案。

书的链接：[Hands-on Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do):

[![book](http://akamaicovers.oreilly.com/images/0636920052289/cat.gif)](http://shop.oreilly.com/product/0636920052289.do)

Simply open the [Jupyter](http://jupyter.org/) notebooks you are interested in:

只需打开你感兴趣的Jupyter notebook就可以了：

* Using [jupyter.org's notebook viewer](http://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/index.ipynb)
    * note: [github.com's notebook viewer](https://github.com/ageron/handson-ml/blob/master/index.ipynb) also works but it is slower and the math formulas are not displayed correctly,
* or by cloning this repository and running Jupyter locally. This option lets you play around with the code. In this case, follow the installation instructions below.

* 使用[jupyter.org的notebook查看器](http://nbviewer.jupyter.org/github/ageron/handson-ml/blob/master/index.ipynb)
    * 注：[github.com的notebook查看器](https://github.com/ageron/handson-ml/blob/master/index.ipynb) 也可以用，但是它很慢，而且数学公式不能正确显示。
* 或者克隆这个repository在本地jupyter运行。这可以让你愉快的和代码玩耍。想在本地查看，请照着下面的安装引导走。

# Installation
# 安装
First, you will need to install [git](https://git-scm.com/), if you don't have it already.

首先，如果你还没有装[git](https://git-scm.com/)，你需要把它装上。

Next, clone this repository by opening a terminal and typing the following commands:

接着，打开终端输入以下命令就可以克隆这个respository。

    $ cd $HOME  # or any other development directory you prefer 或者创建一个你喜欢的其它文件夹
    $ git clone https://github.com/ageron/handson-ml.git
    $ cd handson-ml

If you do not want to install git, you can instead download [master.zip](https://github.com/ageron/handson-ml/archive/master.zip), unzip it, rename the resulting directory to `handson-ml` and move it to your development directory.

如果你实在不想装git，你也可以下载[master.zip](https://github.com/ageron/handson-ml/archive/master.zip)并解压，重命名解压后的文件夹为`handson-ml`，将其移动到你的开发文件夹。

If you want to go through chapter 16 on Reinforcement Learning, you will need to [install OpenAI gym](https://gym.openai.com/docs) and its dependencies for Atari simulations.

如果你想学习第16章强化学习，你需要[安装OpenAI gym](https://gym.openai.com/docs)以及其Atari simulations附件。

If you are familiar with Python and you know how to install Python libraries, go ahead and install the libraries listed in `requirements.txt` and jump to the [Starting Jupyter](#starting-jupyter) section. If you need detailed instructions, please read on.

如果你对python很熟，知道怎么样装python库，直接装好`requirements.txt`里需要安装的库，然后跳到[Starting Jupyter](#starting-jupyter)部分。如果你需要详细的安装方法，请继续往下阅读。

## Python & Required Libraries
## Python及所需的库
Of course, you obviously need Python. Python 2 is already preinstalled on most systems nowadays, and sometimes even Python 3. You can check which version(s) you have by typing the following commands:

当然，你肯定需要Python。python2在大多数系统都是预装好的，有些甚至python3都预装了。你可以输入以下代码查看预装的python版本：

    $ python --version   # for Python 2
    $ python3 --version  # for Python 3

Any Python 3 version should be fine, preferably ≥3.5. If you don't have Python 3, I recommend installing it (Python ≥2.6 should work, but it is deprecated so Python 3 is preferable). To do so, you have several options: on Windows or MacOSX, you can just download it from [python.org](https://www.python.org/downloads/). On MacOSX, you can alternatively use [MacPorts](https://www.macports.org/) or [Homebrew](https://brew.sh/). If you are using Python 3.6 on MacOSX, you need to run the following command to install the `certifi` package of certificates because Python 3.6 on MacOSX has no certificates to validate SSL connections (see this [StackOverflow question](https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)):

所有的python3版本都是可以的，最好版本不低于3.5。如果你没有python3,我建议把它装上（虽然python2也可以，但是不推荐）。要安装，你有几个选择：在windows和macosx下，你可以直接从[python.org](https://www.python.org/downloads/)下载。在macosx下，你也可以使用[MacPorts](https://www.macports.org/)或者[Homebrew]证书，因为python3.6在macosx里没有证书来验证SSL连接（看这个[StackOverflow解答](https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)):

    $ /Applications/Python\ 3.6/Install\ Certificates.command

On Linux, unless you know what you are doing, you should use your system's packaging system. For example, on Debian or Ubuntu, type:

在linux下，除非你知道你在做什么，你应该使用你系统的安装包系统。例如，在Debian或者Ubuntu下，输入：

    $ sudo apt-get update
    $ sudo apt-get install python3

Another option is to download and install [Anaconda](https://www.continuum.io/downloads). This is a package that includes both Python and many scientific libraries. You should prefer the Python 3 version.

还有一个选择就是下载安装[Anaconda](https://www.continuum.io/downloads)。这是一个包含了python和很多科学计算库的包。你应该选择python3版本。

If you choose to use Anaconda, read the next section, or else jump to the [Using pip](#using-pip) section.

如果你选择使用anaconda，接着阅读下面的部分，如果不是就跳到[使用pip](#using-pip)部分。

## Using Anaconda
## 使用Anaconda
When using Anaconda, you can optionally create an isolated Python environment dedicated to this project. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this project), with potentially different libraries and library versions:

当使用Anaconda时，

    $ conda create -n mlbook python=3.5 anaconda
    $ source activate mlbook

This creates a fresh Python 3.5 environment called `mlbook` (you can change the name if you want to), and it activates it. This environment contains all the scientific libraries that come with Anaconda. This includes all the libraries we will need (NumPy, Matplotlib, Pandas, Jupyter and a few others), except for TensorFlow, so let's install it:

    $ conda install -n mlbook -c conda-forge tensorflow

This installs the latest version of TensorFlow available for Anaconda (which is usually *not* the latest TensorFlow version) in the `mlbook` environment (fetching it from the `conda-forge` repository). If you chose not to create an `mlbook` environment, then just remove the `-n mlbook` option.

Next, you can optionally install Jupyter extensions. These are useful to have nice tables of contents in the notebooks, but they are not required.

    $ conda install -n mlbook -c conda-forge jupyter_contrib_nbextensions

You are all set! Next, jump to the [Starting Jupyter](#starting-jupyter) section.

## Using pip 
## 使用pip
If you are not using Anaconda, you need to install several scientific Python libraries that are necessary for this project, in particular NumPy, Matplotlib, Pandas, Jupyter and TensorFlow (and a few others). For this, you can either use Python's integrated packaging system, pip, or you may prefer to use your system's own packaging system (if available, e.g. on Linux, or on MacOSX when using MacPorts or Homebrew). The advantage of using pip is that it is easy to create multiple isolated Python environments with different libraries and different library versions (e.g. one environment for each project). The advantage of using your system's packaging system is that there is less risk of having conflicts between your Python libraries and your system's other packages. Since I have many projects with different library requirements, I prefer to use pip with isolated environments. Moreover, the pip packages are usually the most recent ones available, while Anaconda and system packages often lag behind a bit.

These are the commands you need to type in a terminal if you want to use pip to install the required libraries. Note: in all the following commands, if you chose to use Python 2 rather than Python 3, you must replace `pip3` with `pip`, and `python3` with `python`.

First you need to make sure you have the latest version of pip installed:

    $ pip3 install --user --upgrade pip

The `--user` option will install the latest version of pip only for the current user. If you prefer to install it system wide (i.e. for all users), you must have administrator rights (e.g. use `sudo pip3` instead of `pip3` on Linux), and you should remove the `--user` option. The same is true of the command below that uses the `--user` option.

Next, you can optionally create an isolated environment. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this project), with potentially very different libraries, and different versions:

    $ pip3 install --user --upgrade virtualenv
    $ virtualenv -p `which python3` env

This creates a new directory called `env` in the current directory, containing an isolated Python environment based on Python 3. If you installed multiple versions of Python 3 on your system, you can replace `` `which python3` `` with the path to the Python executable you prefer to use.

Now you must activate this environment. You will need to run this command every time you want to use this environment.

    $ source ./env/bin/activate

On Windows, the command is slightly different:

    $ .\env\Scripts\activate

Next, use pip to install the required python packages. If you are not using virtualenv, you should add the `--user` option (alternatively you could install the libraries system-wide, but this will probably require administrator rights, e.g. using `sudo pip3` instead of `pip3` on Linux).

    $ pip3 install --upgrade -r requirements.txt

Great! You're all set, you just need to start Jupyter now.

## Starting Jupyter
## 开始Jupyter
If you want to use the Jupyter extensions (optional, they are mainly useful to have nice tables of contents), you first need to install them:

如果你想使用Jupyter扩展（可选，他们主要是为了产生更漂亮的表和内容），你首先需要安装它们：

    $ jupyter contrib nbextension install --user

Then you can activate an extension, such as the Table of Contents (2) extension:

然后你可以激活一个扩展，比如Contents (2)扩展的表：

    $ jupyter nbextension enable toc2/main

Okay! You can now start Jupyter, simply type:

okay！你现在可以打开Jupyter了，只需输入：

    $ jupyter notebook

This should open up your browser, and you should see Jupyter's tree view, with the contents of the current directory. If your browser does not open automatically, visit [localhost:8888](http://localhost:8888/tree). Click on `index.ipynb` to get started!

这会打开你的浏览器，你就能看到Jupyter树界面，里面是当前文件夹的内容。如果你的浏览器没有自动打开，访问[localhost:8888](http://localhost:8888/tree)。打开`index.ipynb`开始吧。

Note: you can also visit [http://localhost:8888/nbextensions](http://localhost:8888/nbextensions) to activate and configure Jupyter extensions.

注：你也可以访问[http://localhost:8888/nbextensions](http://localhost:8888/nbextensions)来激活并配置Jupyter扩展。

Congrats! You are ready to learn Machine Learning, hands on!

恭喜！你已经准备好学习机器学习了，动手吧！

# Contributors
# 贡献者
I would like to thank everyone who contributed to this project, either by providing useful feedback, filing issues or submitting Pull Requests. Special thanks go to Steven Bunkley and Ziembla who created the `docker` directory.

我想谢谢每一个为这个项目作出贡献的人，那些给了有用的反馈，解决了问题或者提交的推送请求的人。特别感谢Steven Bunkley和Ziemb创建了`docker`文件夹。
