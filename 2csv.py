import csv
import json
with open('data/train_with_summary.txt', 'r') as fp:
	dict = fp.readlines()
	with open('data/test.csv', 'w', newline = '') as csvfile:
		fieldnames = ['summarization', 'article']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for dict_report in dict:
			#print(json.loads(dict_report))
			writer.writerow(json.loads(dict_report))