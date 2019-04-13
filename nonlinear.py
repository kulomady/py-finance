from scipy.stats import pearsonr, spearmanr
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_general_correlation(df, column_to_split, column_to_correlate, percent_data_to_correlate):
    df=df.sort_values([column_to_split])
    stats=df[column_to_split].describe()
    min=stats['min']
    max=stats['max']
    range=max-min
    step_forward=(percent_data_to_correlate*range)/2
    list_of_corr=[]
    total_data=0

    # menghitung partial correlation
    start=min
    while (1.001*start<max):
        data=df[(df[column_to_split]>=start) & (df[column_to_split]<start+2*step_forward)]
        data_size=len(data)
        total_data=total_data+data_size
        #print('start', start, 'end', start+2*step_forward)
        #print('x',data[column_to_split].values, 'y', data[column_to_correlate].values)
        if data_size>2:
           corr=pearsonr(data[column_to_split].values, data[column_to_correlate].values)
           list_of_corr.append((corr, data_size, (start, start+2*step_forward)))

        start=start+step_forward

    total_corr=0
    total_pvalue=0
    for c in list_of_corr:
        scaling_factor=float(c[1])/float(total_data)
        normalized_corr=c[0][0]*scaling_factor
        print("normalize" , normalized_corr)
        normalized_pvalue=c[0][1]*scaling_factor
        total_corr=total_corr+math.fabs(normalized_corr)
        total_pvalue=total_pvalue+normalized_pvalue

    return total_corr, total_pvalue, list_of_corr


if __name__=='__main__':
    np_x=np.array([11,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])
    np_y=np.power(np_x, 2)
    data={'x':np_x, 'y':np_y}
    df=pd.DataFrame(data)
    df=df.sort_values(['x'])
    print(df.head(20))
    pearson_corr=pearsonr(np_x, np_y)
    spearmanr_corr=spearmanr(np_x, np_y)
    general_corr=compute_general_correlation(df, 'x', 'y', 0.25)

    print('pearson corr', pearson_corr)
    print('spearmanr_corr', spearmanr_corr)
    print('general_corr', general_corr[0], general_corr[1])

    plt.scatter(np_x, np_y)
    plt.show()