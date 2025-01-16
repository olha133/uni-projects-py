import pandas as pd
import copy
import matplotlib.pyplot as plt
from matplotlib import style

def prepare_cv_data():
    df = pd.read_csv("CV_data.csv")
    PSET0 = df[df.Class == '+']
    NSET0 = df[df.Class == '-']
    PSET, NSET = list(), list()
    for _, row in PSET0.iterrows():
        row_list = list(row.iloc[:-1])  # select all values except the last one
        PSET.append(row_list)
    for _, row in NSET0.iterrows():
        row_list = list(row.iloc[:-1])  # select all values except the last one
        NSET.append(row_list)
    
    HSET = [[set(df['Height']),set(df['Hair']),set(df['Eyes'])]]

    return PSET,NSET,HSET

def prepare_covid_data():
    df = pd.read_csv("Covid_data.csv")

    # Visualize the distribution of severity classes
    plt.style.use('seaborn-v0_8-muted')
    # plt.figure(figsize=(4, 4))
    # plt.bar(['Severe', 'None'], df['Severity_None'].value_counts())
    # plt.title('Distribution of severity classes')
    # plt.xlabel('Severity')
    # plt.ylabel('Count')
    # plt.show()

    # plt.figure(figsize=(10, 8))
    # plt.matshow(df.corr(), fignum=1)
    # plt.xticks(range(len(df.columns)), df.columns, rotation='vertical')
    # plt.yticks(range(len(df.columns)), df.columns)
    # plt.colorbar()
    # plt.title('Correlation matrix')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.bar(df.columns[:-2], df.iloc[:, :-2].sum(), color='blue')
    # plt.xticks(rotation='vertical')
    # plt.title('Frequency of symptoms')
    # plt.xlabel('Symptom')
    # plt.ylabel('Count')
    # plt.show()

    #   Visualize the frequency of each symptom by severity class
    # plt.figure(figsize=(10, 6))
    # plt.bar(df.columns[:-2], df[df['Severity_Severe'] == 1].iloc[:, :-2].sum(), label='Severe')
    # plt.bar(df.columns[:-2], df[df['Severity_None'] == 1].iloc[:, :-2].sum(), label='None')
    # plt.xticks(rotation='vertical')
    # plt.title('Frequency of symptoms by severity class')
    # plt.xlabel('Symptom')
    # plt.ylabel('Count')
    # plt.legend()
    # plt.show()

    PSET0 = df[df.Severity_Severe == 1]
    NSET0 = df[df.Severity_None == 1]
    PSET, NSET = list(), list()
    for _, row in PSET0.iterrows():
        row_list = list(row.iloc[:-2])  # select all values except the two last
        PSET.append(row_list)
    for _, row in NSET0.iterrows():
        row_list = list(row.iloc[:-2])  # select all values except the two last
        NSET.append(row_list)
    HSET = [[set(df['Fever']),set(df['Tiredness']),set(df['Dry-Cough']), set(df['Difficulty-in-Breathing']),set(df['Sore-Throat']),set(df['None_Sympton']),set(df['Pains']),set(df['Nasal-Congestion']),set(df['Runny-Nose']),set(df['Diarrhea'])]]
    
    return PSET,NSET,HSET
def specify(H):
    SPECS = []
    for S in H:
        if len(S) > 1:
            for elem in S.copy():
                new_list = copy.deepcopy(H)
                idx = new_list.index(S)
                new_list[idx].discard(elem)
                SPECS.append(new_list)
    return SPECS

def score(S, PSET, NSET):
    M = len(PSET) + len(NSET)
    P = 0
    N = 0
    for row in PSET:
        idx = 0
        while idx < len(row):
            if row[idx] not in S[idx]:
                break
            elif idx == len(row) - 1:
                P += 1
                idx += 1
            else:
                idx += 1
    
    for row in NSET:
        idx = 0
        while idx < len(row):
            if row[idx] not in S[idx]:
                N += 1
                break
            else:
                idx += 1
    score = (P + N) / M
    return score

def is_as_specific(S, C):
    sum1 = 0
    sum2 = 0
    for my_set in S:
        sum1 += len(my_set)
    for my_set in C:
        sum2 += len(my_set)
    return sum1 == sum2

def hgs(PSET, NSET, CLOSED_SET, HSET, beam_size):
    OPEN_SET = list()
    for H in HSET:
        SPECS = specify(H)
        NEW_SET = list()
        for S in SPECS:
            if score(S, PSET, NSET) > score(H, PSET, NSET):
                NEW_SET.append(S)
        if not NEW_SET:
            CLOSED_SET.append(H)
        else:
            for S in NEW_SET:
                OPEN_SET.append(S)

    for C in CLOSED_SET.copy():
        for S in OPEN_SET.copy():
            if is_as_specific(S, C):
                if score(C, PSET, NSET) > score(S, PSET, NSET):
                    OPEN_SET.remove(S)
                else:
                    CLOSED_SET.remove(C)

    if not OPEN_SET:
        return max(CLOSED_SET, key=lambda x: score(x, PSET, NSET))
    else:
        BEST_SET = sorted(OPEN_SET + CLOSED_SET, key=lambda x: score(x, PSET, NSET), reverse=True)[:beam_size]
        CLOSED_SET = [x for x in CLOSED_SET if x in BEST_SET]
        OPEN_SET = [x for x in OPEN_SET if x in BEST_SET]
        return hgs(PSET, NSET, CLOSED_SET, OPEN_SET, beam_size)

def evaluate(result, PSET, NSET):
    TP = TN = FP = FN = 0
    M = len(PSET) + len(NSET)
    for row in PSET:
        idx = 0
        while idx < len(row):
            if row[idx] not in result[idx]:
                FN += 1
                break
            elif idx == len(row) - 1:
                TP += 1
                idx += 1
            else:
                idx += 1
    for row in NSET:
        idx = 0
        while idx < len(row):
            if row[idx] not in result[idx]:
                TN += 1
                break
            elif idx == len(row) - 1:
                FP += 1
                idx += 1
            else:
                idx += 1
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2 * ((precision * recall)/(precision + recall))
    accuracy = (TP+TN)/M
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    errorRate = (FP+FN)/M

    conf_matrix = [[TP, FP], [FN, TN]]

    # # Set the plot size and title
    plt.figure(figsize=(5, 4))
    plt.title('Confusion Matrix')

    # Create a heatmap to display the confusion matrix
    heatmap = plt.pcolor(conf_matrix, cmap='Blues')

    # Add the count of each value in the confusion matrix
    for y in range(2):
        for x in range(2):
            plt.text(x + 0.5, y + 0.5, '{}'.format(conf_matrix[y][x]),
                    horizontalalignment='center',
                    verticalalignment='center')

    # Set the x and y axis labels
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Set the tick labels for the x and y axis
    tick_labels = ['Positive', 'Negative']
    plt.xticks([0.5, 1.5], tick_labels)
    plt.yticks([0.5, 1.5], tick_labels)
    # Add a color bar to the right of the heatmap
    plt.colorbar(heatmap)
    plt.show()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'TP rate', 'TN rate', 'P Predictive Value', 'N Predictive Value','Error Rate']
    values = [round(accuracy, 4), round(precision, 4), round(recall, 4), round(F1,4), round(TPR, 4), round(TNR, 4),\
              round(PPV, 4), round(NPV, 4), round(errorRate, 4)]
    metrics.reverse()
    values.reverse()
    plt.barh(metrics, values)
    for i, v in enumerate(values):
        plt.text(v , i , str(v), color='blue', fontsize = 'small')
    plt.show()


def main():
    PSET,NSET,HSET = prepare_cv_data()

    # PSET,NSET,HSET = prepare_covid_data()
    result = hgs(PSET, NSET, list(), HSET, beam_size = 2)
    evaluate(result, PSET, NSET)
    print(result)
if __name__ == "__main__":
    main()