for in_name in [""]:
    for function in ["multilevel"]:
        for rec in [3]:
            for mode in ["union"]:
                arq_path = f'G:\Mestrado\BD\data\outputs\saida_{function}_0_{mode}_{rec}_d=2{in_name}.out'
                with open(arq_path, 'r') as f:
                    for line in f:
                        if line[0:3] == 'MDS':
                            table_line = f'{line[4:-1]}' + ' & '
                        elif 'Adjusted Rand index: ' in line:
                            length = len('Adjusted Rand index: ')
                            table_line += line[length:-1] + ' & '
                        elif 'Adjusted Mutual Information: ' in line:
                            length = len('Adjusted Mutual Information: ')
                            table_line += line[length:-1] + ' & '
                        elif 'Homogeneity: ' in line:
                            length = len('Homogeneity: ')
                            table_line += line[length:-1] + ' & '
                        elif 'Completeness: ' in line:
                            length = len('Completeness: ')
                            table_line += line[length:-1] + ' & '
                        elif 'V-measure: ' in line:
                            length = len('V-measure: ')
                            table_line += line[length:-1] + ' & '
                        elif 'Fowlkes-Mallows: ' in line:
                            length = len('Fowlkes-Mallows: ')
                            table_line += line[length:-1] + ' \\\\'
                            
                            table_line = table_line.replace('%', '\%')
                            table_line = table_line.replace('_', '\_')
                            print(table_line + '\n' + '\hline')
                        # elif 'Silhouette: ' in line:
                        #     length = len('Silhouette: ')
                        #     table_line += line[length:-1] + ' \\\\'
                        #     print(table_line + '\n' + '\hline')