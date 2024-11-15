import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from HS_reading.lag_brain_plot import lag_process_data
from HS_reading.lag_brain_plot import calcu_median
from HS_reading.lag_brain_plot import data_process
from collections import defaultdict
import itertools
from HS_reading.lag_brain_plot import prepare_dict

def arrow_plot(node_to_remove,rlist,lag_list):
    lag_list=calcu_median(lag_list)
    rlist=calcu_median(rlist)
    lag_list = dict(sorted(lag_list.items()))
    rlist=dict(sorted(rlist.items()))


    for k in lag_list.keys():
        if lag_list[k]:
            temp = lag_list[k]
            rlist_temp = rlist[k]
            node_remove_list=node_to_remove[k]
            G = nx.DiGraph()

            for k1 in temp.keys():
                key_list=k1.split('_')
                if key_list[0] in node_remove_list or key_list[1] in node_remove_list:
                    continue
                else:
                    if temp[k1]<=0:
                        #如果lag<0,代表第一个脑区要提前，则第二个脑区指向第一个脑区，反之亦然
                        G.add_edge(key_list[1], key_list[0], weight=temp[k1],width=rlist_temp[k1]*5)
                    else:
                        G.add_edge(key_list[0], key_list[1], weight=-temp[k1],width=rlist_temp[k1]*5)


            edge_weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
            edge_widths = np.array([d['width'] for _, _, d in G.edges(data=True)])

            pos = nx.spring_layout(G)


            angles = [2 * np.pi * i / len(pos) for i in range(len(pos))]
            pos = {node: (np.cos(angle), np.sin(angle)) for node, angle in zip(pos, angles)}

            fig, ax = plt.subplots(figsize=(12, 12))

            nx.draw_networkx_edges(G, pos, edge_color=edge_weights, edge_cmap=plt.cm.Blues_r,width=edge_widths, arrowstyle='-|>', arrowsize=20, ax=ax)

            nx.draw_networkx_nodes(G,pos=pos, node_color='lightblue',node_size=1000,ax=ax )
            nx.draw_networkx_labels(G, pos=pos, font_size=10, font_color='black', ax=ax)

            norm = plt.Normalize(vmin=np.min(edge_weights), vmax=np.max(edge_weights))
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues_r, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Avg. time of max lag (s)',fontsize=20)
            ax.xaxis.set_label_coords(0.5, -0.15)

            cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=90)
            ax.tick_params(axis='both', which='both', labelsize=6)

            plt.axis('off')
            plt.title(f'{k}',fontsize=50)
            plt.tight_layout()
            plt.show()

if __name__=='__main__':
    mat_path = "E:/vs/python/data_for_code/HSblockdata/elecs/elec_sig"
    node_to_remove={'overt':[],'covert':[],'cue':[]}
    loc_path = "E:/vs/python/data_for_code/HSblockdata/"



    ##origin
    clean_path_1="E:/vs/python/lags"
    HS_list = [44, 45, 47, 48, 50,54 ,71,73,76,78]
    width1 = 0.2
    rlist_1,lag_list_1 = lag_process_data(HS_list,loc_path, mat_path, clean_path_1, node_to_remove)
    arrow_plot(node_to_remove,rlist_1,lag_list_1)

    #new
    clean_path_2 = "E:/vs/python/"
    HS_list_1=[45]
    task_list=['overt']
    wts_max_index,r_value,r2_value,r_value_brain_pair,r2_value_brain_pair, wts_max_index_brain_pair=data_process(clean_path_2,HS_list_1,task_list,loc_path)
    rlist=r_value_brain_pair
    r2list=r2_value_brain_pair
    lag_list=wts_max_index_brain_pair
    arrow_plot(node_to_remove,rlist,lag_list)
    arrow_plot(node_to_remove,r2list,lag_list)

