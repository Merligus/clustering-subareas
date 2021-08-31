comb = []
for cut in [0, 0.2]:
    for only_journals in [0, 1]:
        for year in [0, 2010]:
            if only_journals and cut > 0:
                continue
            in_name = ""
            in_name += f'_only_journals={only_journals}'
            in_name += f'_cut={cut}'
            in_name += f'_year={year}'
            params = ""
            if year > 0:
                params += ' desde ' + str(year) + ','
            if only_journals:
                params += ' só jornais,'
            if cut > 0:
                params += f' cut {cut},'
            comb.append((in_name, params[:-1]))

for in_name, params in comb:
    for function, rec in [("multilevel", 3)]:
        for log in [0]:
            for nan in [0]:
                for mode in ["union", "mean", "max"]:
                    arq_path = f'G:\Mestrado\BD\data\outputs\definitivas\\'
                    # saida_multilevel_0_union_4_d=1_nan=0_only_journals=0_cut=0.2_year=0.out
                    filename = f'saida_{function}_{log}_{mode}_{rec}_d=1_nan={nan}{in_name}.out'
                    with open(arq_path + filename, 'r') as f:
                        for line in f:
                            if line[0:3] == 'INI':
                                table_line = f'{mode},{params}' + ' & '
                            elif 'Adjusted Rand index: ' in line:
                                length = len('Adjusted Rand index: ')
                                table_line += line[length:-1] + ' & '
                            elif 'Adjusted Mutual Information: ' in line:
                                length = len('Adjusted Mutual Information: ')
                                table_line += line[length:-1] + ' & '
                            # elif 'Homogeneity: ' in line:
                            #     length = len('Homogeneity: ')
                            #     table_line += line[length:-1] + ' & '
                            # elif 'Completeness: ' in line:
                            #     length = len('Completeness: ')
                            #     table_line += line[length:-1] + ' & '
                            elif 'V-measure: ' in line:
                                length = len('V-measure: ')
                                table_line += line[length:-1] + ' & '
                            elif 'Fowlkes-Mallows: ' in line:
                                length = len('Fowlkes-Mallows: ')
                                table_line += line[length:-1] + ' & '
                                
                                # table_line = table_line.replace('%', '\%')
                                # table_line = table_line.replace('_', '\_')
                                # print(table_line + '\n' + '\hline')    
                            elif 'Silhouette: ' in line:
                                length = len('Silhouette: ')
                                table_line += line[length:-1] + ' \\\\'

                                table_line = table_line.replace('%', '\%')
                                table_line = table_line.replace('_', '\_')
                                print(table_line + '\n' + '\hline')