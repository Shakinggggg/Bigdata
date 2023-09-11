import pandas as pd
import re


def load_data(path):
    Data = pd.read_csv(path, encoding='utf-8')
    return Data


def salary_process(ori_data, salary_type):
    salary = re.split('-|·', ori_data['薪资待遇'])
    # print(ori_data['薪资待遇'])
    salary_low, salary_high = salary[0], "".join(list(filter(str.isdigit, salary[1])))
    if salary_type == 0:
        if '月' in ori_data['薪资待遇']:
            salary_low = int(salary_low) // 1000
            salary_high = int(salary_high) // 1000
            return (salary_low + salary_high) / 2
        elif '天' in ori_data['薪资待遇']:
            salary_low = int(salary_low) * 30 // 1000
            salary_high = int(salary_high) * 30 // 1000
            return (salary_low + salary_high) / 2
        else:
            return (int(salary_low) + int(salary_high)) / 2
    else:
        if '薪' in salary[-1]:
            return int("".join(list(filter(str.isdigit, salary[-1]))))
        else:
            return 12


def pos_process(ori_data, is_exp):
    pos = re.split('，', ori_data['岗位详情'])
    if is_exp:
        if '周' in pos[0]:
            return '经验不限'
        return pos[0]
    else:
        return pos[-1]


def loc_process(ori_data):
    loc = re.split('·', ori_data['岗位地区'])
    return loc[0]


def welfare_process(ori_data):
    if ori_data['公司福利'] != ori_data['公司福利']:
        return 0
    else:
        wel = re.split('，', ori_data['公司福利'])
        return len(wel)


def skill_process(ori_data):
    skill = re.split('，', ori_data['岗位需要技能'])
    return len(skill)
    # print(skill)

def company_process(ori_data, pro_type):
    company = re.split('，', ori_data['公司规模'])
    # print(company)
    if pro_type == 0:
        # print(company[0])
        return company[0]
    elif pro_type == 1:
        if len(company) == 2:
            if '人' in company[1]:
                return '未知'
            else:
                return company[1]
        else:
            return company[1]
    else:
        if len(company) == 2:
            if '0' in company[1]:
                return company[1]
            else:
                return '未知'
        else:
            return company[-1]

def data_stats(data):
    df = data
    df['salary1'] = df.apply(lambda r: salary_process(r, 1), axis=1)
    df['salary2'] = df.apply(lambda r: salary_process(r, 0), axis=1)
    df['学历要求'] = df.apply(lambda r: pos_process(r, 0), axis=1)
    df['工作经验要求'] = df.apply(lambda r: pos_process(r, 1), axis=1)
    df['岗位地区'] = df.apply(lambda r: loc_process(r), axis=1)
    # df.drop(columns='岗位详情', inplace=True)
    df['福利数量'] = df.apply(lambda r:welfare_process(r), axis=1)
    df['需求技能数量'] = df.apply(lambda r: skill_process(r), axis=1)
    df['公司类型'] = df.apply(lambda r: company_process(r, 0), axis=1)
    df['融资情况'] = df.apply(lambda r: company_process(r, 1), axis=1)
    df['公司人数'] = df.apply(lambda r: company_process(r, 2), axis=1)
    df.drop(columns='公司规模', inplace=True)
    df.to_csv('dealed_data3.csv')
    # print(df)


if __name__ == '__main__':
    data = load_data('job2(1).csv')
    data_stats(data)


