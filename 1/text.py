# encoding=utf-8
import jieba

seg_list = jieba.cut("我来到北京清华大学, 哈哈哈. 其次, 这很好玩.")
print(list(seg_list))


