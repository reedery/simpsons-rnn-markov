import pandas as pd
data = pd.read_csv("/Users/jamesledoux/Downloads/script_lines_db.csv")

textfile = ""
episode_number = 1
for index, row in data.iterrows():
	textfile += row['raw_text'] 
	textfile += '\n'

out_file = open("all_scripts.txt", "w")
out_file.write(textfile)
out_file.close()
