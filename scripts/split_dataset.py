import os

import perplexity
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from collections import defaultdict
import pandas as pd


def split_file(file_name, split_string):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    question_number = 0
    student_number = 0
    gpt_number = 0
    current_file = None
    question_folder = None
    for line in lines:
        if split_string in line:
            question_number += 1
            question_folder = f"../tmp/question_{question_number}"
            if not os.path.exists(question_folder):
                os.makedirs(question_folder)
            student_number = 0
            gpt_number = 0
            current_file = None
        elif "Student" in line:
            if current_file:
                current_file.close()
            student_number += 1
            current_file_name = f"student_{question_number}_{student_number}"
            current_file = open(os.path.join(question_folder, current_file_name), 'w')
        elif "GPT" in line:
            if current_file:
                current_file.close()
            gpt_number += 1
            current_file_name = f"gpt_{question_number}_{gpt_number}"
            current_file = open(os.path.join(question_folder, current_file_name), 'w')
        elif current_file:
            current_file.write(line)


def normalize_text(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    with open(file_name, 'w') as file:
        for line in lines:
            line = line.strip()
            line = ' '.join(line.split())
            while len(line) > 120:
                line_split = line[:120].rfind(' ')
                file.write(line[:line_split] + '\n')
                line = line[line_split + 1:]
            if line:
                file.write(line + '\n')


def normalize_folder(folder_name):
    for root, dirs, files in os.walk(folder_name):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            normalize_text(full_path)


def plot_dict(d):
    sns.barplot(x=list(d.keys()), y=list(d.values()))
    plt.xlabel('Keys')
    plt.ylabel('Values')
    plt.title('Plot of dictionary values')
    plt.xticks(rotation=60)
    plt.show()
    # plt.savefig("plot.svg", format='svg')


def calculate_average(data, keys_to_group):
    grouped_data = defaultdict(list)
    for key, value in data.items():
        if key in keys_to_group:
            grouped_data[key].append(value)
    avg_data = {key: sum(values) / len(values) for key, values in grouped_data.items()}
    return avg_data


def print_perplexities(folder_name):
    column_names = ["type", "question", "num", "ppl"]
    l = []
    for root, dirs, files in os.walk(folder_name):
        index = 0
        for file_name in files:
            full_path = os.path.join(root, file_name)
            column_values = file_name.split('_')
            print(f"File: {full_path}, file_name: {file_name}")
            with open(full_path, 'r') as f:
                ppl = perplexity.perplexity(f.read())
                column_values.append(ppl)
                l.append(column_values)

    ppls_per_file = pd.DataFrame(list(l), columns=column_names)
    save_dict(ppls_per_file, "ppl_per_file.dataframe")


def save_dict(d, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(d, file)


def load_dict(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


# split_file("../dataset/mihalis.txt", "Question ")
# normalize_folder("../tmp")
# print_perplexities("../tmp")

df2 = load_dict("ppl_per_file.dataframe")
df2.question = df2.question.astype(int)
print(f"Average: {df2.groupby(['type'])['ppl'].mean()}")

df2.sort_values(by=['question', 'num'], inplace=True)
group_by_type_student = df2.loc[df2['type'] == 'student'].groupby(['question'])['ppl'].mean()
group_by_type_gpt = df2.loc[df2['type'] == 'gpt'].groupby(['question'])['ppl'].mean()
group_by_question_by_type = df2.groupby(['question', 'type']).mean('ppl').unstack('type')

figure, axis = plt.subplots(2, 2)
figure.tight_layout(pad=2.0)

f1 = axis[0, 1]
f1.plot(group_by_type_student)
f1.set_title('Group by Type = Student', fontdict={'fontsize': 8})
f1.set_xlabel('question_num')
f1.set_ylabel('mean perplexity')


f2 = axis[0, 1]
f2.plot(group_by_type_gpt)
f2.set_title('Group by Type = GPT', fontdict={'fontsize': 8})
f2.set_xlabel('question_num')
f2.set_ylabel('mean perplexity')

f3 = axis[1, 0]
f3.plot(group_by_question_by_type)
f3.set_title('Group by Type = GPT', fontdict={'fontsize': 8})
f3.set_xlabel('question_num')
f3.set_ylabel('mean perplexity')

print(df2.head())

f4 = axis[1, 1]
f4.plot(df2['ppl'], df2[['question', 'type', 'num']])
f4.set_xlabel('question_num')
f4.set_ylabel('perplexity')

plt.show()
