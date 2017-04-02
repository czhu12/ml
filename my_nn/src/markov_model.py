import markovify
import csv

with open('text/tweets.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    all_tweets = []
    for row in spamreader:
        handle, tweet, is_retweet = row[1], row[2], row[3]
        if handle == 'realDonaldTrump' and is_retweet == 'False':
            tweet = tweet.decode(encoding='UTF-8')
            words = tweet.split(' ')
            words = filter(lambda x: 'http' not in x, words)
            tweet = ' '.join(words)
            print(handle, tweet)
            all_tweets.append(tweet)

    tweets = '\n'.join(all_tweets)
    text_model = markovify.Text(tweets)
    for i in range(100):
        print(text_model.make_short_sentence(140))


# Get raw text as string.
#with open("text/donald.txt") as f:
#    text = f.read()
#
## Build the model.
#text_model = markovify.Text(text)
#
## Print three randomly-generated sentences of no more than 140 characters
#for i in range(3):
#    print(text_model.make_short_sentence(140))
#
