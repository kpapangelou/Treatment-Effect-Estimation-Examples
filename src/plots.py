import matplotlib.pyplot as plt

def create_boxplot(data,names_leg,xtick_lab,x_lab=None,y_lab=None,ref=None,legend_loc=None):

    num_methods = len(data)
    num_simulations = data[0].shape[0]

    cmap = plt.get_cmap('tab10')
    fig,ax = plt.subplots()
    k=0
    last_k=0
    plt_xticks = []
    for i in range(num_simulations):
        plt_data = []
        plt_pos = []
        for j in range(num_methods):
            plt_data.append(data[j][i,:])
            plt_pos.append(k)
            k+=1
        bp=ax.boxplot(plt_data,positions=plt_pos)
        for j in range(num_methods):
            plt.setp(bp['boxes'][j], color=cmap(j))
            plt.setp(bp['medians'][j], color=cmap(j))
        plt_xticks.append(last_k+(num_methods-1)/2)
        k+=1
        last_k = k
    if ref:
        ax.axhline(ref,color='k')
    ax.set_xticks(plt_xticks)
    ax.set_xticklabels(xtick_lab)
    if x_lab:
        ax.set_xlabel(x_lab)
    if y_lab:
        ax.set_ylabel(y_lab)
    plt_leg = []
    for i in range(num_methods):
        plt_leg.append(bp["boxes"][i])
    if legend_loc is not None:
        ax.legend(plt_leg, names_leg,loc=legend_loc)
    else:
        ax.legend(plt_leg, names_leg)