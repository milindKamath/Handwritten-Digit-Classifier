from bs4 import BeautifulSoup
import os, sys, csv
import random


def read_in(path):
    values = {}
    for data in os.listdir(path):
        with open(path + '/' + data) as fp:
            soup = BeautifulSoup(fp, features='lxml')
            tag = soup.find('annotation', {'type': 'UI'})
            if tag is not None:
                ui = tag.text
                values[ui] = data
    return values


def ground_truth_dict(gt):
    ground_truth = {}
    with open(gt, mode='r') as infile:
        reader = csv.reader(infile, quotechar=None)
        ground_truth = {rows[0]: rows[1] for rows in reader}
    return ground_truth


def split_data(path, ground_truth, name):
    gt = ground_truth_dict(ground_truth)
    data = read_in(path)
    classes = {}
    for key in gt.keys():
        if gt[key] not in classes.keys():
            classes[gt[key]] = [key]
        else:
            classes[gt[key]].append(key)
    test = []
    train = []
    for klass in classes.keys():
        all_class = classes[klass]
        total = len(all_class)
        for i in range(int(total * .7)):
            idx = random.choice(all_class)
            train.append(data[idx])
            all_class.remove(idx)
        for leftover in all_class:
            test.append((data[leftover]))
    with open('train-' + name + '.txt', 'w+') as t:
        for file in train:
            t.write(file + '\n')
    with open('test-' + name + '.txt', 'w+') as t:
        for file in test:
            t.write(file + '\n')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Ussage: python training_test [path] [ground truth] [name]")
    else:
        path = sys.argv[1]
        gt = sys.argv[2]
        name = sys.argv[3]
        split_data(path, gt, name)