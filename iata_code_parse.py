def iata_parse(filename:str):
    import pandas as pd
    file = pd.read_csv(filename)

    iata_dict={}
    for i in range(0, len(file.header), 2):
        iata_dict.update({file.header.values[i] : file.header.values[i+1]})
    df = pd.DataFrame(iata_dict.items())
    df.to_csv('iata_code_captials.csv', index=False)

iata_parse("cities.txt")
