import pandas as pd
import string
import gensim
from gensim.parsing.preprocessing import preprocess_string

def data_reader(data, id_var, description_var, drop_missing=True):
    id = data[id_var].values
    descriptions = data[description_var].values
    data_frame = {'id': id, 'description': descriptions}
    df = pd.DataFrame(data_frame)
    if drop_missing:
        df = df.dropna()
        df.reset_index(inplace=True, drop=True)
    return df

def preprocess_text(df, description_var, filters=None, printable_only=True, remove_empty_list_observations=True):
    tokens = []
    printable = set(string.printable)
    if filters is None:
        for i in range(len(df[description_var])):
            if printable_only:
                desc = ''.join(filter(lambda x: x in printable, str(df[description_var][i])))
            else:
                desc = str(df[description_var][i])
            tokens.append(preprocess_string(desc))
    else:
        for i in range(len(df[description_var])):
            if printable_only:
                desc = ''.join(filter(lambda x: x in printable, str(df[description_var][i])))
            else:
                desc = str(df[description_var][i])
            tokens.append(preprocess_string(desc, filters=filters))
    df['tokens'] = tokens
    if remove_empty_list_observations:
        df = df[~df.tokens.str.len().eq(0)]
        df.reset_index(inplace=True, drop=True)
    return df
