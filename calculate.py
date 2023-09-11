from data_deal import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from wordcloud import WordCloud
import matplotlib
# 中文乱码解决方法
plt.rcParams['font.family'] = ['Arial Unicode MS','Microsoft YaHei','SimHei','sans-serif']
plt.rcParams['axes.unicode_minus'] = False
font_path = "C:/Windows/Fonts/msyh.ttc"
# 薪资待遇 岗位地区 salary1	学历要求	工作经验要求	福利数量	需求技能数量	融资情况	人数
weights = [0.22, 0.06, 0.08, 0.08, 0.08, 0.11, 0.09, 0.12, 0.16]


def salary_month_process(ori_data, max_s, min_s):
    return (ori_data['salary2'] - min_s) / (max_s - min_s)


def loc_process(ori_data, loc_dict, max_num, min_num):
    return (loc_dict[ori_data['岗位地区']] - min_num) / (max_num - min_num)


def welfare_process(ori_data, max_num, min_num):
    return (ori_data['福利数量'] - min_num) / (max_num - min_num)


def sal_process(ori_data, max_num, min_num):
    return (ori_data['salary1'] - min_num) / (max_num - min_num)


def edu_process(ori_data, edu_dict, max_num, min_num):
    return (edu_dict[ori_data['学历要求']] - min_num) / (max_num - min_num)


def exp_process(ori_data, exp_dict, max_num, min_num):
    return (exp_dict[ori_data['工作经验要求']] - min_num) / (max_num - min_num)


def skill_process(ori_data, max_num, min_num):
    return (ori_data['需求技能数量'] - min_num) / (max_num - min_num)


def rongzi_process(ori_data, rongzi_dict, max_num, min_num):
    return (rongzi_dict[ori_data['融资情况']] - min_num) / (max_num - min_num)


def size_process(ori_data, size_dict, max_num, min_num):
    return (size_dict[ori_data['公司人数']] - min_num) / (max_num - min_num)


def cal_score(ori_data):
    return (ori_data['salary2'] * weights[0] + ori_data['岗位地区'] * weights[1] +
            ori_data['salary1'] * weights[2] + ori_data['学历要求'] * weights[3] +
            ori_data['工作经验要求'] * weights[4] + ori_data['福利数量'] * weights[5] +
            ori_data['需求技能数量'] * weights[6] + ori_data['融资情况'] * weights[7] +
            ori_data['公司人数'] * weights[8])


def score_normalize(ori_data, max_score, min_score):
    return (ori_data['score'] - min_score) / (max_score - min_score)


def graph_pie(ori_data):
    pie_dict = dict(ori_data['岗位地区'].value_counts())
    num_top_sections = 9
    sorted_data = sorted(pie_dict.items(), key=lambda x: x[1], reverse=True)
    top_sections = sorted_data[:num_top_sections]
    remaining_sections = sorted_data[num_top_sections:]
    remaining_size = sum(v for k, v in remaining_sections)
    top_sections.append(('其他', str(remaining_size)))
    labels = [section[0] for section in top_sections]
    sizes = [section[1] for section in top_sections]
    plt.figure(figsize=(6, 6))  # 设置图形大小（可根据需要调整）
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)  # autopct参数显示百分比，startangle参数设置起始角度
    # 添加标题
    plt.title('各城市岗位数量占比')
    # 显示图形
    plt.show()


def calculate(ori_data):
    sava_df = copy.deepcopy(ori_data)
    df = ori_data
    max_salary_month, min_salary_month = max(list(df['salary2'])), min(list(df['salary2']))
    df['salary2'] = df.apply(lambda r: salary_month_process(r, max_salary_month, min_salary_month), axis=1)

    loc_dict = df['岗位地区'].value_counts().to_dict()
    max_loc, min_loc = max(list(df['岗位地区'].value_counts())), min(list(df['岗位地区'].value_counts()))
    df['岗位地区'] = df.apply(lambda r: loc_process(r, loc_dict, max_loc, min_loc), axis=1)

    max_wel, min_wel = max(list(df['福利数量'])), min(list(df['福利数量']))
    df['福利数量'] = df.apply(lambda r: welfare_process(r, max_wel, min_wel), axis=1)

    max_sal, min_sal = max(list(df['salary1'])), min(list(df['salary1']))
    df['salary1'] = df.apply(lambda r: sal_process(r, max_sal, min_sal), axis=1)

    edu_dict = {'学历不限': 8, '初中及以下': 7, '中专/中技': 6, '高中': 5, '大专': 4, '本科': 3, '硕士': 2, '博士': 1}
    df['学历要求'] = df.apply(lambda r: edu_process(r, edu_dict, 8, 1), axis=1)

    exp_dict = {'经验不限': 10, '在校/应届': 9, '应届生': 9, '1年以内': 8, '1-3年': 7, '3-5年': 6, '5-10年': 5, '10年以上': 4}
    df['工作经验要求'] = df.apply(lambda r: exp_process(r, exp_dict, 10, 4), axis=1)

    max_skill, min_skill = max(list(df['需求技能数量'])), min(list(df['需求技能数量']))
    df['需求技能数量'] = df.apply(lambda r: skill_process(r, max_skill, min_skill), axis=1)

    rongzi_dict = {'已上市': 10, 'D轮及以上': 9, 'C轮': 8, 'B轮': 7, 'A轮': 6, '天使轮': 5, '未融资': 4, '未知': 4, '不需要融资': 4}
    df['融资情况'] = df.apply(lambda r: rongzi_process(r, rongzi_dict, 10, 4), axis=1)

    size_dict = {'10000人以上': 10, '1000-9999人': 9, '500-999人': 8, '100-499人': 7, '20-99人': 6, '0-20人': 5, '未知': 4}
    df['公司人数'] = df.apply(lambda r: size_process(r, size_dict, 10, 4), axis=1)

    df.drop(columns=df.columns[0], axis=1, inplace=True)
    df['score'] = df.apply(lambda r: cal_score(r), axis=1)
    df_max_score, df_min_score = max(list(df['score'])), min(list(df['score']))
    df['score'] = df.apply(lambda r: score_normalize(r, df_max_score, df_min_score), axis=1)
    # print(df)
    # df.to_csv('test_score.csv')
    sava_df['score'] = df.apply(lambda r: cal_score(r), axis=1)
    max_score, min_score = max(list(sava_df['score'])), min(list(sava_df['score']))
    sava_df['score'] = sava_df.apply(lambda r: score_normalize(r, max_score, min_score), axis=1)
    sava_df.drop(columns=df.columns[0], axis=1, inplace=True)
    # 饼图
    # graph_pie(sava_df)
    # print(sava_df)

    # sns.kdeplot(list(sava_df['score']), color='blue')
    # plt.title('Scoring Distribution Plot')
    # plt.xlabel('Score')
    # plt.ylabel('Percentage')
    # plt.show()

    # plt.scatter(list(sava_df['salary2']), list(sava_df['score']), marker='o', color='b')
    # plt.show()

    # text_data = " ".join(list(sava_df['岗位名称']))
    # wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(text_data)
    # # 绘制词云图
    # plt.figure(figsize=(10, 5))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')  # 隐藏坐标轴
    # plt.title('岗位名称词云图')

    # 显示词云图
    # plt.show()
    # print(max(list(sava_df['score'])), min(list(sava_df['score'])))
    # print(df)
    # sava_df.to_csv('score.csv')


if __name__ == '__main__':
    data = load_data('dealed_data3.csv')
    calculate(data)