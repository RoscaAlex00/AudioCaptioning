import sys
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df1 = pd.read_csv(sys.argv[1], encoding='unicode_escape', header=0)
df2 = pd.read_csv(sys.argv[2], header=0)

df = pd.DataFrame()
df.insert(0, "file_name", df1['file_name'])
df.insert(1, "keywords", df1['keywords'])
df.insert(2, "caption_1", df2['caption_1'])
df.insert(3, "caption_2", df2['caption_2'])
df.insert(4, "caption_3", df2['caption_3'])
df.insert(5, "caption_4", df2['caption_4'])
df.insert(6, "caption_5", df2['caption_5'])

output_df = pd.DataFrame(columns=['filename', 'Total_keywords', 'caption1', 'caption2',
                                    'caption3', 'caption4', 'caption5', 'conf'])

for idx, row in df.iterrows():
    keywords = []
    keywords = row['keywords'].split(';')
    keywords_in_caption1 = 0
    keywords_in_caption2 = 0
    keywords_in_caption3 = 0
    keywords_in_caption4 = 0
    keywords_in_caption5 = 0
    for keyword in keywords:
        if keyword in row['caption_1']:
            keywords_in_caption1 += 1
        if keyword in row['caption_2']:
            keywords_in_caption2 += 1
        if keyword in row['caption_3']:
            keywords_in_caption3 += 1
        if keyword in row['caption_4']:
            keywords_in_caption4 += 1
        if keyword in row['caption_5']:
            keywords_in_caption5 += 1
    conf = (keywords_in_caption1 + keywords_in_caption2 + keywords_in_caption3 + keywords_in_caption4 + keywords_in_caption5)/(len(keywords) * 5)
    new_row = { 
                'filename': row['file_name'],
                'Total_keywords':len(keywords),
                'caption1':keywords_in_caption1,
                'caption2':keywords_in_caption2,
                'caption3':keywords_in_caption3,
                'caption4':keywords_in_caption4,
                'caption5':keywords_in_caption5,
                'conf': conf
              }
    output_df = output_df.append(new_row, ignore_index=True)
print(output_df)
output_df.to_csv('keyword_caption_relation.csv', index=False)
print("Mean confidence for keywords being in captions: ", output_df['conf'].mean())