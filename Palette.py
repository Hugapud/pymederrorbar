"""
调色盘.

asd
"""

import colorsys
import random
from typing import Any, Iterable, Union

class Palette:
    """
    调色盘.
    ===
    
    Python>=3.9.0\n
    生成区分度较大的多种颜色
    """

    __digit = list(map(str, range(10))) + list("ABCDEF")

    @staticmethod
    def __gen_n_hls_colors(num: int) -> list[tuple]:
        """生成HLS色彩值."""
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            _hlsc = (h / 360.0, l / 100.0, s / 100.0)
            hls_colors.append(_hlsc)
            i += step

        return hls_colors

    @staticmethod
    def RGB_colors(num: int) -> list[tuple]:
        """生成若干组RGB色彩.

        Return:[(R,G,B),...]
        """
        rgb_colors = []
        if num < 1:
            return rgb_colors
        hls_colors = Palette.__gen_n_hls_colors(num)
        for hlsc in hls_colors:
            _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
            rgb = (int(_r * 255.0), int(_g * 255.0), int(_b * 255.0))
            rgb_colors.append(rgb)

        return rgb_colors

    @staticmethod
    def color_transform(value:Union[tuple,str])->Union[tuple,str]:
        """色彩表示格式转换.

        Return: 传入value若为(R,G,B)则转换为#FFFFFF的HTML-like色彩；传入为#FFFFFF则转换为(R,G,B)
        """
        if isinstance(value, tuple):
            string = '#'
            for i in value:
                a1 = i // 16
                a2 = i % 16
                string +=Palette.__digit[a1] + Palette.__digit[a2]
            return string
        elif isinstance(value, str):
            a1 = Palette.__digit.index(value[1]) * 16 + Palette.__digit.index(value[2])
            a2 = Palette.__digit.index(value[3]) * 16 + Palette.__digit.index(value[4])
            a3 = Palette.__digit.index(value[5]) * 16 + Palette.__digit.index(value[6])
            return (a1, a2, a3)
    
    @staticmethod
    def __to_html_colors(values:Iterable[tuple])->Any:
        """转换[(R,G,B),...]为['#FFFFFF',...]."""
        return [Palette.color_transform(tc) for tc in values]

    @staticmethod
    def HTML_colors(num:int)->list[str]:
        """生成若干组HTML-like色彩.

        Return:['#FFFFFF',...]
        """
        colors=Palette.RGB_colors(num)
        return Palette.__to_html_colors(colors)