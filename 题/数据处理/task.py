import json
import pandas as pd

def document_merge() -> dict:
    data = {}
    #完善代码
    with open("/home/project/2022_february.json","r") as f:
        data["february"] = json.load(f)["february"]

    january = pd.read_excel("/home/project/2022_january.xlsx",skiprows=4).iloc[:,2:]
    january.index = january["date"].dt.strftime("%-m-%-d")
    data["january"] = january.iloc[:,1:].to_dict("index")

    may = pd.read_html("/home/project/2022_may.html",header=0)[0]
    may.index = may["date"]
    data["may"] = may.iloc[:,1:].to_dict("index")
    
    return data

document_merge()