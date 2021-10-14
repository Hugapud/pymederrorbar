"""
绘制箱线图
=====
Copyright 2021 Hugapud
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-----------
mpl_configs配置图形参数
drawing_errorbar()是绘图的主要函数
data_process()是数据检查函数
data_parser()是输入文件解析函数
data_analyse()是统计分析用函数

"""
from Palette import Palette
from typing import Sequence, Union
import statistics as stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

mpl_configs = {
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'font.serif': ['SimSun'],
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'axes.titlesize': 28,
    'axes.titleweight': 'bold',
    'axes.titlepad': 20
}

matplotlib.rcParams.update(mpl_configs)


def drawing_errorbar(labels: list[str], data: np.ndarray, error: np.ndarray, *args,
                     title=None, xlabel: str = 'x轴', ylabel: str = 'y轴', xticks: list[str] = None,
                     xgroup_sep: bool = False, legend_title: Union[str, None] = None, marker='s',
                     capsize: Union[int, float] = 5, markersize: Union[int, float] = 10,
                     dfactor_x: float = 0.1, interval=2, clrs=None, legend_ncol=1,
                     **kwargs) -> None:
    """绘制聚类多序列误差图.

    ===
    该函数用于绘制多序列误差图，
    x轴是条件变量，隐藏；y轴是百分比显示
    Parameters
    ---
    labels : array-like str
        数据聚类的标签(default [])
    data : np.ndarray
        聚类中的数值列表组(default [[]])
    error : np.ndarray
        聚类中的数值偏差组，即±值(default [[]])
    title : Text
        图表的大标题(default None,不显示标题)
    xlabel : str
        x轴标题
    ylabel : str
        y轴标题
    xticks : list[str]
        x轴聚类的标签组(default None,无标签)
    xgroup_sep : bool
        是否显示x轴聚类的分隔线，True--显示，False--不显示(default False)
    legend_title : str
        图例标题(default None,无标题)
    marker : str
        数据点样式，样式字符串
    markersize : int or float
        数据点大小
    dfactor_x : float
        x轴数据坐标缩放比例(default 0.1)
    interval : int
        每聚类末尾留出的padding栏数量，默认为2
    clrs : color array-like
        指定数据类别的图形着色(default None,随机着色)，颜色代码接受map-name或HTML-like，详见matplotlib
    legend_ncol : int
        图例的列数(default 1)
    """
    nx_cata = data_process(
        labels, data, error)  # 数据检查labels,data,error的长度应该保持一致
    n_cata: int = len(labels)

    # 调色选项
    if not clrs or len(clrs) != n_cata:
        # clrs = Palette.HTML_colors(n_cata)
        clrs = ['#66CDAA', '#00FA9A', '#FA8072', '#9370DB', '#FF4500',
                '#1E90FF', '#3CB371', '#FFA500', '#C71585', '#DC143C']

    # 设置坐标轴的绘图属性
    ax: plt.Axes = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))  # 这里固定了x轴的位置，因此x轴不会随拖动变化
    # ax.invert_yaxis()  # 翻转y轴
    # 设置x刻度标签到顶部
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    # 配置x轴
    plt.xlabel(xlabel)
    if not xticks:
        plt.xticks([])
    else:
        xticks_pos = [dfactor_x*(x*(n_cata+interval)+(n_cata+1)/2)
                      for x in range(0, nx_cata)]
        plt.xticks(xticks_pos, xticks[:len(xticks_pos)])
        plt.tick_params('x', color='none')
        plt.tick_params(axis='both', **kwargs)
    # 添加聚类分区线
    if xgroup_sep:
        for i in range(1, nx_cata):
            ax.axvline(dfactor_x*(i*n_cata+(i-0.5)*interval+0.5),
                       linestyle='--', linewidth=1)
    # 配置y轴
    # ylim = np.max(np.max(data+error))*1.05
    # plt.ylim((-ylim, ylim))
    plt.ylabel(ylabel)
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda data, pos: f'{data:.0%}'))
    # 设置轴系的标题
    plt.title(title)
    # 存储errorbar对象
    cats = []
    for i, label, dat, err, clr in zip(range(n_cata), labels, data, error, clrs):
        # 列表生成式可否换为元组加速？
        # x = [dfactor_x*(j*(n_cata+interval) + i+1) for j in range(nx_cata)]
        x = [dfactor_x*j for j in range(i+1, nx_cata *
                                        (n_cata+interval) + i+1, n_cata+interval)]
        cat = plt.errorbar(x, dat, yerr=err, fmt=' ', marker=marker, markersize=markersize, color=clr,
                           capsize=capsize)
        cats.append(cat)
    plt.legend(loc='lower right', handles=cats, labels=labels,
               title=legend_title, mode=None,
               ncol=legend_ncol, title_fontsize='x-large',
               columnspacing=0.5)
    plt.show()


def data_process(labels: Sequence, data: np.ndarray, error: np.ndarray) -> int:
    """
    检查数据一致性.

    TODO：清除无效数据
    Return: X轴的数据分类数
    """
    #
    if(False):
        raise ArithmeticError('数据Size不一致')
    return 3

# 返回多个值的注解怎么做？def data_parser()->list[str],np.ndarray,np.ndarray?:


def data_parser(path: str) -> dict:
    """
    数据导入方法.
    
    Params: 参数
    Return: dict--{BookName:BookContent{X0:{X1:[Ym...]}}}
    """
    wb = pd.read_excel(str(path), engine='openpyxl', sheet_name=None)
    dfs: dict[str, dict] = {}
    # sn--扫描方向.st--数据表
    for sn, st in wb.items():
        sheet = {}
        dfs[sn] = sheet
        col_names: list[str] = list(st.columns)
        for row in st.itertuples(index=False):
            row_data: dict[str, list] = {}
            # 自变量名
            sheet[row[0]] = row_data
            for i in range(1, len(row)):
                # 过滤pandas.nan
                if pd.notna(row[i]):
                    dname = col_names[i][0:2]
                    layer = row_data.get(dname)
                    if not layer:
                        row_data[dname] = []
                        layer = row_data.get(dname)
                    layer.append(row[i])
    return dfs


def data_analyse(diagram: dict, standard: dict):
    """
    数据分析方法.

    Params 参数列表
    Return 
    """
    labels = []
    BMD_avg_errors = []
    BMD_error_sds = []
    for label, row_data in diagram.items():
        avg_errors = []
        error_sds = []
        labels.append(label)
        BMD_avg_errors.append(avg_errors)
        BMD_error_sds.append(error_sds)
        for bone, darray in row_data.items():
            stan = standard[bone]
            # 测量平均值的偏差
            avg_errors.append((stats.mean(darray)-stan)/stan)
            # 测量偏差的样本标准差
            error_sds.append(stats.stdev(map(lambda x: (stan-x)/stan, darray)))
    return labels, BMD_avg_errors, BMD_error_sds


if __name__ == '__main__':
    datatables = data_parser(r'E:\Source\Python\pymederrorbar\data.xlsx')
    # 标准dict
    stands = {'L1': 50.2, 'L2': 100.6, 'L3': 199.2}
    labels, data, error = data_analyse(datatables['Axial-5mm-30'], stands)
    a = np.array(data)
    b = np.array(error)
    drawing_errorbar(labels, a, b, markersize=3,
                     xlabel=r'Nominal HA concentration ($\mathrm{mg/cm^3}$)',
                     ylabel='Mean deviation of the true HA concentration',
                     xgroup_sep=True, xticks=list(['L1 50.2', 'L2 100.6', 'L3 199.2']),
                     legend_ncol=1)
