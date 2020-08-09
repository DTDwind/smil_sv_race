#-*- coding: utf-8 -*-
import math

score_path = "/share/nas165/chunliang/voxceleb_trainer/eval_test/eval_sc_lab.txt"
lines = []

def reliability(percent, score):
    sc_lab = []
    with open(score_path) as scorepath:
        while True:
            line = scorepath.readline()
            if (not line):  # or (len(all_scores)==1000)
                break
            sc_lab.append([line.split()[0], line.split()[1]])
    scorepath.close()

    sc_lab = sorted(sc_lab, key=lambda x:x[0])
    get_num = math.floor(len(sc_lab) * ((1-percent) / 2))  # 左右各取幾個錯誤
    left = get_num
    right = get_num
    # print(get_num)
    if get_num > 421:  # 只有421+423個辨識錯誤
        print("Please increase accuracy percent.")
        result = "ERROR"
    else:
        left_score = sc_lab[0][0]
        right_score = sc_lab[-1][0]
        for i in range(len(sc_lab)):
            if sc_lab[i][1] == "1":
                left = left - 1
            if left == 0:
                left_score = sc_lab[i+1][0]
                break
        for i in range(len(sc_lab)):
            if sc_lab[len(sc_lab)-i-1][1] == "0":
                right = right - 1
            if right == 0:
                right_score = sc_lab[len(sc_lab)-i-2][0]
                break
        # print(left_score + " " + right_score)
        if float(left_score) >= score >= float(right_score):
            result = "TRUE"  # 分數在範圍內
        else:
            result = "FALSE"
    return result


print(reliability(0.99, -1))  # 844/37720 = 0.022 請用0.978以上的準確率
