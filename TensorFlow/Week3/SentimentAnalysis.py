import csv
import re
import preprocessor as p

if __name__ == '__main__':
    output = open("clean_tweets_non_stem.tsv", "w")
    with open("training_cleaned.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            text = row[5].lower()
            label = row[0]
            text = text.replace("'","")
            text = p.clean(text)
            tokens = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", text).split()
            tokens = [t for t in tokens if len(t) > 1]
            if len(tokens) < 3:
                continue
            text = ' '.join(tokens)
            output.write(text + "\t" + label + "\n")
    output.close()
