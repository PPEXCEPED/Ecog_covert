import scipy.io as scio
def classification(HS_list,clear_path):
    total={}
    for HS in HS_list:
        temp = scio.loadmat(clear_path+f"elecs/warped/HS{HS}_elecs_all_warped.mat")
        classified_data={}
        anatomy = temp['anatomy']
        # elecmatrix = temp['elecmatrix']
        length=len(anatomy)
        for i in range(length):
            region=anatomy[i][3][0]
            # electrode=elecmatrix[i]
            if region in classified_data:
                classified_data[region].append(i)
            else:
                classified_data[region]=[i]

        #合并部分解剖标签
        new_classified_data = {}
        for key, value in classified_data.items():
            if "middlefrontal" in key:
                # 如果键中包含 "middlefrontalgyrus" 字符串
                if "middlefrontal" in new_classified_data:
                    # 如果新字典中已经存在 "middlefrontalgyrus" 这个键
                    new_classified_data["middlefrontal"].extend(value)
                else:
                    # 如果新字典中还没有 "middlefrontalgyrus" 这个键
                    new_classified_data["middlefrontal"] = value
            elif "bankssts" in key:
                # 如果键中包含 "middlefrontalgyrus" 字符串
                if "middletemporal" in new_classified_data:
                    # 如果新字典中已经存在 "middlefrontalgyrus" 这个键
                    new_classified_data["middletemporal"].extend(value)
                else:
                    # 如果新字典中还没有 "middlefrontalgyrus" 这个键
                    new_classified_data["middletemporal"] = value
            else:
                # 如果键中不包含 "middlefrontalgyrus" 字符串
                new_classified_data[key] = value

        total[f'{HS}']=new_classified_data

    return total

if __name__=='__main__':
    HS_list=[44,45,47,48,50,54,71,73,76,78]
    clear_path="/Users/zhaozehao/Desktop/reading task/"
    print(classification(HS_list,clear_path))