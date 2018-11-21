import urllib.request, json 
import csv

def getJsonFromUrl(url):
    with urllib.request.urlopen(finalUrl) as url:
        data = json.loads(url.read().decode())
        return data

def getListFromCsv(filePath):
    with open(filePath, 'r', encoding='utf-8') as readFile:
        reader = csv.reader(readFile)
        lines = list(reader)
        return lines
